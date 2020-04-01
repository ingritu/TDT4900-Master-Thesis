from pathlib import Path

import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]


"""
caption_id patterns: 
- image_id # caption_num -> str[0].isaplha() # .isdecimal  // correct order
- image_id caption_num #  // problematic if not space separated
- # image_id caption_num  // problematic if not space separated
- # caption_num image_id  // common error
- caption_num image_id #
- caption_num # image_id
"""


def contains_caption_id(string):
    original_str = string
    if string.strip().isdecimal():  # matches index
        return False

    string = repair_captions([string])[0]
    string = string.replace(",", "")
    string = string.replace(" ", "")
    if string.strip().isalpha():  # matches caption
        return False

    if '_' in original_str and '#' in original_str:
        return True
    return False


def is_image_id(string):
    # example COCO_val2014_000000477949
    # example COCO_train2014_000000467654
    base_val = "COCO_val2014_000"
    base_train = "COCO_train2014_0"
    length_val = len("COCO_val2014_000000477949")
    length_train = len("COCO_train2014_000000467654")
    str_len = len(string)
    base = string[:16]
    rest = string[16:]
    base_bool = base == base_val or base == base_train
    length_bool = str_len == length_val or str_len == length_train
    return base_bool and length_bool and rest.isdecimal()


def end_of_caption_idx(string):
    # exploit that all caption_ids seem to end will always end with a
    # number or a # symbol
    idx = -1
    str_len = len(string)
    # string = COCO_val2014_000000581683 # 585508 a wedding cake
    # that is high 5 layers and flowers on it.
    splits = string.split(" ")
    sp_len = len(splits)
    while sp_len + idx > 0:
        item = splits[idx].strip()
        if item == "#" or is_image_id(item):
            cap_id = " ".join(splits[:sp_len + idx + 1]).strip()
            return len(cap_id) - str_len - 1
        elif item.isdecimal():
            # could be part of caption or is cap_num
            # look ahead
            next_item = splits[idx - 1].strip()
            if next_item == "#" or is_image_id(next_item):
                # this is the actual end of caption_id
                cap_id = " ".join(splits[:sp_len + idx + 1]).strip()
                return len(cap_id) - str_len - 1
        idx -= 1
    return None


def missing_comma(string):
    new = string
    new = new.replace(" ", "")
    new = new.replace("\t", "")
    new = new.replace("\u200b", "")
    return "," not in new[0:]


def repair_file(file):
    """
    Repairs translated files.

    Some files after being translated have som errors in them. This is
    problematic because files will be translated twice. So if they are not
    repaired then the errors just gets worse. In particular we need
    caption_id and caption to be clearly separated and to keep caption_id
    uncorrupted.

    Errors that are not repaired are detected so that you could easily
    repair the error yourself.

    Errors include:
    - no "," after index.
    - caption_num and image_name switch order in caption_id.
        - # caption_num image_name.
    - weired symbols in captions like '"', '«' and '»'.
    - caption_id: image_name # caption_num.
    - extra "," in the middle of caption_num.
    - spaces after ','.
    - period instead of comma after index.

    Parameters
    ----------
    file : Path or str.
    """
    print("Repairing file:", str(file))
    file = Path(file)
    with open(file, 'r') as f:
        lines = f.readlines()
    #print(lines[1])
    lines = lines[1:]  # remove labels
    lines = repair_captions(lines, cap_bool=False)
    lines = [line for line in lines if len(line) > 1]
    tmp_lines = []
    recent = 0
    print("repair indices")
    for line in lines:
        string, recent = repair_index(line, recent)
        #print(recent, type(recent))
        tmp_lines.append(string)
    lines = tmp_lines
    print("indices repaired")

    labels = ",caption_id,caption\n"
    indices, caption_ids, captions = [], [], []
    for line in lines:
        line = line.split(",")  # most lines are still comma separated
        print(line)
        indices.append(line[0].replace(' ', '').strip())
        line = line[1:]  # remove index
        caption_id = ''
        evaluate = line[0]
        if contains_caption_id(evaluate):
            end_idx = end_of_caption_idx(evaluate)
            if end_idx is not None:
                caption_id = evaluate[:len(evaluate) + end_idx + 1]
                caption_ids.append(caption_id)
                rest = evaluate[len(evaluate) + end_idx + 1:]
                # join rest with the rest of line
                caption = ', '.join([rest] + line[1:])
                captions.append(caption)
        else:
            # join with the next one and assume this enough
            evaluate = ' '.join([evaluate] + [line[1]])
            if contains_caption_id(evaluate):
                end_idx = end_of_caption_idx(evaluate)
                if end_idx is not None:
                    caption_id = evaluate[:len(evaluate) + end_idx + 1]
                    caption_ids.append(caption_id)
                    rest = evaluate[len(evaluate) + end_idx + 1:]
                    # join rest with the rest of line
                    caption = ', '.join([rest] + line[2:])
                    captions.append(caption)
        assert caption_id != '', "wrong in caption_id fetching"

    for index in indices:
        if not index.isdecimal():
            print('index', index)
    assert sum([index.isdecimal() for index in indices]) == len(indices), \
        "Indices are not correct."

    print("repair caption_id ...")
    caption_ids = repair_caption_ids(caption_ids)  # should be done
    print("repair captions ...")
    captions = repair_captions(captions)  # should be done

    # prepare to write to file
    print("write to file ...")
    out_str = labels
    for index, caption_id, caption in zip(indices, caption_ids, captions):
        if "," in caption[:3]:
            caption = caption.replace(",", "", 1)
            caption = caption.strip()
        out_str += index + ',' + caption_id + ',' + caption + '\n'

    outfile = str(file).replace('.txt', '')  # remove txt
    outfile += '.repaired.txt'
    with open(outfile, 'w') as out_f:
        out_f.write(out_str)
    print("DONE!!!")


def repair_index(string, recent_index=0):
    tmp_index = -1
    if missing_comma(string):
        # find end of index
        # find idx of space
        space_idx = string.find(" ", 0, 10)
        if space_idx != -1:
            tmp_index = string[0:space_idx]
            try:
                tmp_index = int(tmp_index)
            except ValueError:
                print(string)
            if recent_index == 0:
                # just accept the index regardless
                string = str(tmp_index) + "," + string[space_idx:]
                tmp_index = tmp_index
            elif tmp_index == recent_index + 1:
                # found index
                string = str(tmp_index) + "," + string[space_idx:]
                tmp_index = tmp_index
            else:
                print("does not match recent index", recent_index)
                print(string)
                new = string
                new = new.replace(" ", "")
                new = new.replace("\u200b", "")
                print(new[0:8])
                exit()
        else:
            print("did not find a space in substring:", string[:10])
            print(string)
            exit()
    else:
        splits = string.split(",")
        tmp_index = splits[0].strip()
        tmp_index = tmp_index.replace(" ", "")
        tmp_index = tmp_index.replace("\u200b", "")
        tmp_index = int(tmp_index)
        if recent_index != 0:
            # else the string is already correct
            if tmp_index != recent_index + 1:
                # try joining with next item
                tmp_index = ''.join([item.strip() for item in splits[:2]])
                tmp_index = tmp_index.replace(" ", "")
                try:
                    tmp_index = int(tmp_index)
                except ValueError:
                    print(string)
                if tmp_index == recent_index + 1:
                    # modify string
                    # matches recent now
                    string = str(tmp_index) + "," + ','.join(splits[2:])
                else:
                    print("index error")
                    print(string)
                    exit()
            else:
                # modify string
                # matches recent
                string = str(tmp_index) + "," + ','.join(splits[1:])

    return string, tmp_index


def repair_caption_ids(caption_ids):
    out = []
    for idx, caption_id in enumerate(caption_ids):
        items = caption_id.split(' ')
        items = [item.strip() for item in items if len(item) > 0]
        items = [item.replace("\u200b", "") for item in items]
        if len(items) != 3:
            cap_num_space_last = \
                sum([itm.strip().isdecimal() for itm in items[2:]]) == \
                len(items[2:])
            cap_num_space_first = \
                sum([itm.strip().isdecimal() for itm in items[0:2]]) == \
                len(items[0:2])
            cap_num_space_middle = \
                sum([itm.strip().isdecimal() for itm in items[1:3]]) == \
                len(items[1:3])
            if cap_num_space_last:
                # join the numbers
                cap_num = ''.join([itm.strip() for itm in items[2:]])
                items = items[:2] + [cap_num]
            elif cap_num_space_first:
                # join the numbers
                cap_num = ''.join([itm.strip() for itm in items[:2]])
                items = [cap_num] + items[2:]
            elif cap_num_space_middle:
                # join the numbers
                cap_num = ''.join([itm.strip() for itm in items[1:3]])
                items = [items[0]] + [cap_num] + items[3:]
            else:
                print(idx, items)
                exit()
        if items[0] == '#':
            # caption_id starts with #
            if items[1][0].isalpha():
                # caption_id: # image_name caption_num
                repaired = [items[1]] + [items[0]] + [items[2]]
                repaired = ''.join(repaired)
            else:
                # caption_id: # caption_num image_name
                # most common error
                repaired = [items[2]] + [items[0]] + [items[1]]
                repaired = ''.join(repaired)
        elif items[0][0].isalpha():
            # image_name first
            if items[1].isalnum():
                # caption_id: image_name caption_num #
                repaired = [items[0]] + [items[2]] + [items[1]]
                repaired = ''.join(repaired)
            else:
                # caption_id: image_name # caption_num
                # it is already correct
                repaired = ''.join(items)
        else:
            # caption_num first
            if items[1] == '#':
                # caption_id: caption_num # image_name
                repaired = [items[2]] + [items[1]] + [items[0]]
                repaired = ''.join(repaired)
            else:
                # caption_id: caption_num image_name #
                repaired = [items[1]] + [items[2]] + [items[0]]
                repaired = ''.join(repaired)
        repaired = repaired.replace(',', '').strip()
        out.append(repaired)
    return out


def repair_captions(captions, cap_bool=True):
    out = []
    for cap in captions:
        if not cap_bool:
            if '.' in cap[:10]:
                cap = cap.replace('.', ',', 1)
        cap = cap.replace('"', '')
        cap = cap.replace('«', '')
        cap = cap.replace('»', '')
        if cap_bool:
            cap = cap[1:]
        cap = cap.strip()
        out.append(cap)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                        help='name of file to repair without the .txt')
    parser.add_argument('--postfix', type=str, default='.en.fr')
    args = vars(parser.parse_args())
    file_ = args['file'] + args['postfix'] + '.txt'
    repair_files_dir = ROOT_PATH.joinpath('data', 'external', 'c5')

    repair_file(repair_files_dir.joinpath(file_))
