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


def end_of_caption_id(string):
    # exploit that all caption_ids seem to end will always end with a
    # number or a # symbol
    idx = -1
    str_len = len(string)
    while str_len + idx > 0:
        c = string[idx]
        if c == "#" or c.isdecimal():
            return idx
        idx -= 1
    return None


def repair_file(file):
    """
    Repairs translated files.

    Some files after being translated have som errors in them. This is
    problematic because files will be translated twice. So if they are not
    repaired then the errors just gets worse. In particular we need
    caption_id and caption to be clearly separated and to keep caption_id
    uncorrupted.

    Errors include:
    - caption_num and image_name switch order in caption_id.
        - # caption_num image_name
    - weired symbols in captions like '"', '«' and '»'
    - caption_id: image_name # caption_num
    - spaces after ','

    Parameters
    ----------
    file : Path or str.
    """
    print("Repairing file:", str(file))
    file = Path(file)
    with open(file, 'r') as f:
        lines = f.readlines()
    print(lines[1])
    lines = lines[1:]  # remove labels
    lines = repair_captions(lines, cap_bool=False)
    lines = [line for line in lines if len(line) > 0]
    labels = ",caption_id,caption\n"
    indices, caption_ids, captions = [], [], []
    for line in lines:
        line = line.split(",")  # most lines are still comma separated
        print(line)
        indices.append(line[0])
        line = line[1:]
        print(line)
        caption_id = ''
        evaluate = line[0]
        if contains_caption_id(evaluate):
            end_idx = end_of_caption_id(evaluate)
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
                end_idx = end_of_caption_id(evaluate)
                if end_idx is not None:
                    caption_id = evaluate[:len(evaluate) + end_idx + 1]
                    caption_ids.append(caption_id)
                    rest = evaluate[len(evaluate) + end_idx + 1:]
                    # join rest with the rest of line
                    caption = ', '.join([rest] + line[2:])
                    captions.append(caption)
        assert caption_id != '', "wrong in caption_id fetching"

    assert sum([index.isdecimal() for index in indices]) == len(indices), \
        "Indices are not correct."

    caption_ids = repair_caption_ids(caption_ids)  # should be done
    captions = repair_captions(captions)  # should be done

    # prepare to write to file
    out_str = labels
    for index, caption_id, caption in zip(indices, caption_ids, captions):
        out_str += index + ',' + caption_id + ',' + caption + '\n'

    outfile = str(file).replace('.txt', '')  # remove txt
    outfile += '.repaired.txt'
    with open(outfile, 'w') as out_f:
        out_f.write(out_str)


def repair_caption_ids(caption_ids):
    out = []
    for caption_id in caption_ids:
        items = caption_id.split(' ')
        items = [item.strip() for item in items if len(item) > 0]
        if len(items) != 3:
            print(items)
        #assert len(items) == 3, "some other error also exists."
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
                        help='name of file to repair without the .en.fr.txt')
    args = vars(parser.parse_args())
    file_ = args['file'] + '.en.fr.txt'
    repair_files_dir = ROOT_PATH.joinpath('data', 'external', 'c5')

    repair_file(repair_files_dir.joinpath(file_))
