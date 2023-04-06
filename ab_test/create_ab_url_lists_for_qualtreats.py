from collections import defaultdict

def load_urls(txtfile):
    with open(txtfile, 'r') as f:
        urls = f.readlines()
    urls = [line.rstrip('\n') for line in urls]
    return urls

####################################################
# load url lists
sac_grapheme_url_list = load_urls("url_lists/SAC-grapheme-input.txt")
sac_speechcode_scot_url_list = load_urls("url_lists/SAC-speechcode-input-scot-fem.txt")
sac_speechcode_us_url_list = load_urls("url_lists/SAC-speechcode-input-us-fem.txt")
vanillatts_grapheme_url_list = load_urls("url_lists/vanillatts-grapheme-input.txt")

####################################################
# sanity check that url lists are as expected

# lengths
assert len(sac_grapheme_url_list) == len(sac_speechcode_scot_url_list) == len(sac_speechcode_us_url_list) == len(vanillatts_grapheme_url_list)

# same words and word order
def extract_word(url):
    """extract target word from url"""
    if "vanillatts" in url:
        w = url.split("vanillatts")[-1].lstrip('-').split('.wav')[0]
    elif '<' in url:
        # speech codes
        w = url.split("<")[-1].split('>')[0]
    else:
        # sac model graphemes
        w = url.split('how is ')[-1].split(' pronounced')[0]
    return w

for a1, a2, a3, a4 in zip(sac_grapheme_url_list, sac_speechcode_us_url_list, sac_speechcode_scot_url_list, vanillatts_grapheme_url_list):
    # print(extract_word(a1), extract_word(a2), extract_word(a3), extract_word(a4))
    assert extract_word(a1) == extract_word(a2) == extract_word(a3) == extract_word(a4)

####################################################
# split url lists into subsets

# 78 total words, we want to create 6 tests each with 13 words from each pair of conditions. using latin square to keep balanced
ranges = [
    (0,13),
    (13,26),
    (26,39),
    (39,52),
    (52,65),
    (65,78),
]

def list_to_dict_using_ranges(l):
    """split list according to ranges"""
    dict_with_ranges = defaultdict(list)
    for start, end in ranges:
        for i in range(start, end):
            dict_with_ranges[(start, end)].append(l[i])
    return dict_with_ranges

sac_grapheme_url_dict = list_to_dict_using_ranges(sac_grapheme_url_list)
sac_speechcode_scot_url_dict = list_to_dict_using_ranges(sac_speechcode_scot_url_list)
sac_speechcode_us_url_dict = list_to_dict_using_ranges(sac_speechcode_us_url_list)
vanillatts_grapheme_url_dict = list_to_dict_using_ranges(vanillatts_grapheme_url_list)

# for key, value in sac_grapheme_url_dict.items():
#     print()
#     print(key, value)

"""system pairs to letter mapping
                 V   sac_graph   sac_us   sac_scot_f
vanilla          
sac-graph        A
sac_us_fem       B       C
sac_scot_fem     D       E          F
"""

####################################################
# create paired url stimuli sets according to latin square

letter2conditionpair = {
    'A': (sac_grapheme_url_dict, vanillatts_grapheme_url_dict),
    'B': (sac_speechcode_us_url_dict, vanillatts_grapheme_url_dict),
    'C': (sac_speechcode_us_url_dict, sac_grapheme_url_dict),
    'D': (sac_speechcode_scot_url_dict, vanillatts_grapheme_url_dict),
    'E': (sac_speechcode_scot_url_dict, sac_grapheme_url_dict),
    'F': (sac_speechcode_scot_url_dict, sac_speechcode_us_url_dict),
}

test_tups = [
    (1, ('A','B','F','C','E','D')),
    (2, ('B','C','A','D','F','E')),
    (3, ('C','D','B','E','A','F')),
    (4, ('D','E','C','F','B','A')),
    (5, ('E','F','D','A','C','B')),
    (6, ('F','A','E','B','D','C')),
]

for test_num, test_pair_letters in test_tups:
    print(f"\ntest === {test_num} ===")
    all_a_urls = []
    all_b_urls = []
    # each test pair only gets a certain number of stimuli according to ranges
    # e.g. test 1, A gets 1-13, B gets 14-26, F gets 27-39
    for range, test_pair_letter in zip(ranges, test_pair_letters):
        # extract urls according to range for the test_pair
        condition_a_url_dict, condition_b_url_dict =  letter2conditionpair[test_pair_letter]
        a_urls = condition_a_url_dict[range]
        b_urls = condition_b_url_dict[range]

        all_a_urls.extend(a_urls)
        all_b_urls.extend(b_urls)

    for a_url, b_url in zip(all_a_urls, all_b_urls):
        # print(f"test {test_num}",a_url,b_url)
        assert a_url != b_url
        assert extract_word(a_url) == extract_word(b_url)

    # save to disk
    def write_url_list_to_disk(test_num, urls, a_or_b):
        lines = []
        for url in urls:
            lines.append(f"test{test_num}_{a_or_b} {url}")
        with open(f"url_lists_ab/ab-urls-test{test_num}_{a_or_b}.txt", 'w') as f:
            f.write("\n".join(lines))

    write_url_list_to_disk(test_num, all_a_urls, "a")
    write_url_list_to_disk(test_num, all_b_urls, "b")
