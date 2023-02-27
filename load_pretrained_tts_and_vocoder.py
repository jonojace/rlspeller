import torch
import argparse
from fastpitch import models as fastpitch_model
import sys

def parse_args(parser):
    """Parse commandline arguments"""
    parser.add_argument('-o', '--chkpt-save-dir', type=str, # required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    train = parser.add_argument_group('training setup')
    train.add_argument('--cuda', action='store_true',
                    help='Enable GPU training')
    train.add_argument('--num-cpus', type=int, default=1,
                    help='Num of cpus on node. Used to optimise number of dataloader workers during training.')
    train.add_argument('--batch-size', type=int, default=16,
                    help='Batchsize (this is divided by number of GPUs if running Data Distributed Parallel Training)')
    train.add_argument('--val-num-to-gen', type=int, default=32,
                    help='Number of samples to generate in validation (determines how many samples show up in wandb')
    train.add_argument('--seed', type=int, default=1337,
                    help='Seed for PyTorch random number generators')
    train.add_argument('--grad-accumulation', type=int, default=1,
                    help='Training steps to accumulate gradients for')
    train.add_argument('--epochs', type=int, default=100,  # required=True,
                    help='Number of total epochs to run')
    train.add_argument('--max-iters-per-epoch', type=int, default=None,
                    help='Number of total batches to iterate through each epoch (reduce this to small number to quickly test whole training loop)')
    train.add_argument('--epochs-per-checkpoint', type=int, default=10,
                    help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                    help='Checkpoint path to resume train')
    train.add_argument('--resume', action='store_true',
                    help='Resume train from the last available checkpoint')
    train.add_argument('--val-log-interval', type=int, default=5,
                    help='How often to generate melspecs/audio for respellings and log to wandb')
    train.add_argument('--speech-length-penalty-training', action='store_true',
                    help='Whether or not to encourage model to output similar length outputs\
                    as the ground truth. Idea from V2C: Visual Voice Cloning (Chen et al. 2021)')
    train.add_argument('--skip-before-train-loop-validation', action='store_true',
                    help='Skip running validation before model training begins (mostly for speeding up testing of actual training loop)')
    train.add_argument('--avg-loss-by-speech_lens', action='store_true',
                    help='Average the softdtw loss according to number of timesteps in predicted sequence')
    train.add_argument('--softdtw-temp', type=float, default=0.01,
                    help='How hard/soft to make min operation. Minimum is recovered by setting this to 0.')
    train.add_argument('--softdtw-bandwidth', type=int, default=120,
                    help='Bandwidth for pruning paths in alignment matrix when calculating SoftDTW')
    train.add_argument('--dist-func', type=str, default="l1",
                    help='What distance function to use in softdtw loss calculation')
    train.add_argument('--cross-entropy-loss', action='store_true',
                    help='Whether to ONLY train the model with cross entropy using grapheme based targets'
                            'will not use fastpitch TTS acoustic loss')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb', choices=['adam', 'lamb'],
                    help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                    help='Learning rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                    help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                    help='Clip threshold for gradients')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                    help='Number of steps for lr warmup')

    arch = parser.add_argument_group('architecture')
    arch.add_argument('--dropout-inputs', type=float, default=0.0,
                    help='Dropout prob to apply to sum of word embeddings '
                        'and positional encodings')
    arch.add_argument('--dropout-layers', type=float, default=0.1,
                    help='Dropout prob to apply to each layer of Tranformer')
    arch.add_argument('--d-model', type=int, default=128,
                    help='Hidden dimension of tranformer')
    arch.add_argument('--d-feedforward', type=int, default=512,
                    help='Hidden dimension of tranformer')
    arch.add_argument('--num-layers', type=int, default=4,
                    help='Number of layers for transformer')
    arch.add_argument('--nheads', type=int, default=4,
                    help='Hidden dimension of tranformer')
    arch.add_argument('--embedding-dim', type=int, default=384, # 384 is default value for fastpitch embedding table
                    help='Hidden dimension of grapheme embedding table')
    arch.add_argument('--pretrained-embedding-table', action='store_true',
                    help='Whether or not to initialise embedding table from fastpitchs')
    arch.add_argument('--freeze-embedding-table', action='store_true',
                    help='Whether or not to allow grapheme embedding input table for EncoderRespeller to be updated.')
    arch.add_argument('--gumbel-temp', nargs=3, type=float, default=(2, 0.5, 0.999995),
                    help='Temperature annealling parameters for Gumbel-Softmax (start, end, decay)')
    arch.add_argument('--no-src-key-padding-mask', dest='src_key_padding_mask', action='store_false',
                    help='Whether or not to provide padding attention mask to Transformer Encoder layers')
    arch.add_argument('--respelling-len-modifier', type=int, default=0, # 384 is default value for fastpitch embedding table
                    help='How many letters to remove from or add to original spelling.')
    arch.add_argument('--use-respelling-len-embeddings', action='store_true', # 384 is default value for fastpitch embedding table
                    help='Whether or not to incorporate to respeller input additional embeddings that indicate how long'
                        'the desired respelling should be.')
    arch.add_argument('--concat-pos-encoding', action='store_true',
                    help='Whether or not to concatenate pos encodings to inputs or sum')
    arch.add_argument('--pos-encoding-dim', type=int, default=128,
                    help='Dim of positional encoding module')
    arch.add_argument('--dont-only-predict-alpha', dest='only_predict_alpha', action='store_false',
                    help='Allow gumbel softmax to predict whitespace, padding, and other punctuation symbols')

    pretrained_tts = parser.add_argument_group('pretrained tts model')
    # pretrained_tts.add_argument('--fastpitch-with-mas', type=bool, default=True,
    #                   help='Whether or not fastpitch was trained with Monotonic Alignment Search (MAS)')
    pretrained_tts.add_argument('--fastpitch-chkpt', type=str, # required=True,
                                help='Path to pretrained fastpitch checkpoint')
    pretrained_tts.add_argument('--input-type', type=str, default='char',
                                choices=['char', 'phone', 'pf', 'unit'],
                                help='Input symbols used, either char (text), phone, pf '
                                    '(phonological feature vectors) or unit (quantized acoustic '
                                    'representation IDs)')
    pretrained_tts.add_argument('--symbol-set', type=str, default='english_basic_lowercase',
                                help='Define symbol set for input sequences. For quantized '
                                    'unit inputs, pass the size of the vocabulary.')
    pretrained_tts.add_argument('--n-speakers', type=int, default=1,
                                help='Condition on speaker, value > 1 enables trainable '
                                    'speaker embeddings.')
    # pretrained_tts.add_argument('--use-sepconv', type=bool, default=True,
    #                   help='Use depthwise separable convolutions')

    audio = parser.add_argument_group('log generated audio')
    audio.add_argument('--hifigan', type=str,
                    default='/home/s1785140/pretrained_models/hifigan/ljspeech/LJ_V1/generator_v1',
                    help='Path to HiFi-GAN audio checkpoint')
    audio.add_argument('--hifigan-config', type=str,
                    default='/home/s1785140/pretrained_models/hifigan/ljspeech/LJ_V1/config.json',
                    help='Path to HiFi-GAN audio config file')
    audio.add_argument('--sampling-rate', type=int, default=22050,
                    help='Sampling rate for output audio')
    audio.add_argument('--hop-length', type=int, default=256,
                    help='STFT hop length for estimating audio length from mel size')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--wordaligned-speechreps', type=str,
                    default='/home/s1785140/data/ljspeech_fastpitch/wordaligned_mels',
                    help='Path to directory of wordaligned speechreps/mels. Inside are folders\
                    each named as a wordtype and containing tensors of word aligned speechreps for each example')
    data.add_argument('--train-wordlist', type=str,
                    default='/home/s1785140/data/ljspeech_fastpitch/respeller_train_words.json',
                    help='Path to words that are used to train respeller')
    data.add_argument('--val-wordlist', type=str,
                    default='/home/s1785140/data/ljspeech_fastpitch/respeller_dev_words.json',
                    help='Path to words that are used to report validation metrics for respeller')
    data.add_argument('--max-examples-per-wordtype', type=int, default=1,
                    help='Path to words that are used to report validation metrics for respeller')
    data.add_argument('--text-cleaners', type=str, nargs='+',
                    default=(),
                    help='What text cleaners to apply to text in order to preproces it before'
                        'its fed to respeller.')

    cond = parser.add_argument_group('conditioning on additional attributes')
    dist = parser.add_argument_group('distributed training setup')

    wandb_logging = parser.add_argument_group('wandb logging')
    data.add_argument('--wandb-project-name', type=str,
                    # required=True,
                    help="The name of the wandb project to add this experiment's logs to")
    wandb_logging.add_argument('--keys-to-add-to-exp-name', type=str, nargs='+',
                    default=(),
                    help='Command line arguments that we add their info to the wandb experiment name')

    return parser

def parse_args_for_model_loading():
    gamma = 0.1
    lr = 0.1 # def for lamb optimizer is 0.001
    dist_metric = 'l1'
    exp_name = f"test_development"
    fastpitch_chkpt = '/home/s1785140/respeller/fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt'

    # imitate CLAs
    sys.argv = [
        'train.py',
        '--chkpt-save-dir', f'/home/s1785140/respeller/exps/{exp_name}', 
        '--fastpitch-chkpt', fastpitch_chkpt,
        '--input-type', 'char',
        '--symbol-set', 'english_pad_lowercase_nopunc',
        # '--text-cleaners', 'lowercase_no_punc',
        '--use-mas',
        '--cuda',
        '--n-speakers', '1',
        '--use-sepconv',
        # '--add-spaces',
        # '--eos-symbol', '$',
        '--batch-size', '2',
        '--val-num-to-gen', '2',
        '--softdtw-temp', str(gamma),
        '--dist-func', dist_metric,
        '--learning-rate', str(lr),
                
        # # NB for development!
        '--epochs', '2', # NB for development!
        '--val-log-interval', '1', # NB for development!
        '--max-iters-per-epoch', '5', # NB for development!
    ]

    parser = argparse.ArgumentParser(description='PyTorch Respeller Training', allow_abbrev=False)
    parser = parse_args(parser)
    args, _unk_args = parser.parse_known_args()

    parser = fastpitch_model.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        print(f'WARNING - Invalid options {unk_args}')

    if args.cuda:
        args.num_gpus = torch.cuda.device_count()
        args.distributed_run = args.num_gpus > 1
        args.batch_size = int(args.batch_size / args.num_gpus)
    else:
        args.distributed_run = False

    device = torch.device('cuda' if args.cuda else 'cpu')

    return args, device

def load_vocoder(hifigan_checkpoint_path):
    """Load HiFi-GAN vocoder from checkpoint"""
    args, device = parse_args_for_model_loading()
    args.hifigan = hifigan_checkpoint_path
    checkpoint_data = torch.load(args.hifigan)
    vocoder_config = fastpitch_model.get_model_config('HiFi-GAN', args)
    vocoder = fastpitch_model.get_model('HiFi-GAN', vocoder_config, device)
    vocoder.load_state_dict(checkpoint_data['generator'])
    vocoder.remove_weight_norm()
    vocoder.eval()
    return vocoder

def load_checkpoint(args, model, filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    return model

def load_pretrained_fastpitch(fastpitch_checkpoint_path):
    args, device = parse_args_for_model_loading()
    args.fastpitch_chkpt = fastpitch_checkpoint_path
    model_config = fastpitch_model.get_model_config('FastPitch', args)
    fastpitch = fastpitch_model.get_model('FastPitch', model_config, device, forward_is_infer=True)
    load_checkpoint(args, fastpitch, args.fastpitch_chkpt)
    n_symbols = fastpitch.encoder.word_emb.weight.size(0)
    grapheme_embedding_dim = fastpitch.encoder.word_emb.weight.size(1)
    return fastpitch, n_symbols, grapheme_embedding_dim, model_config
