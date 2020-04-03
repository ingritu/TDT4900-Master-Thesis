from pathlib import Path
import pandas as pd


ROOT_PATH = Path(__file__).absolute().parents[2]

external_path = ROOT_PATH.joinpath('data', 'external')

postfix = '.en.fr.repaired.fr.en.repaired.txt'

directory = external_path.joinpath('c5')

data = {
    "idx": [],
    "image_id": [],
    "image_name": [],
    "caption_id": [],
    "caption": []
}

# collect all data
for i in range(71):
    filename = directory.joinpath('c5_' + str(i) + postfix)
    # filename is the absolute path to file.
    print(filename)
    with open(Path(filename), "r") as file:
        file_data = file.readlines()
    file_data = file_data[1:]  # remove header
    file_data = [line.split(",") for line in file_data]

    # add indices
    indices = [int(line[0]) for line in file_data]
    data["idx"].extend(indices)

    # add captions
    for line in file_data:
        if not line[2].isdecimal():
            cap = ' '.join(line[2:]).strip()
            data['caption'].append(cap)
        else:
            # in case there is an extra ',' before the caption
            cap = ' '.join(line[3:]).strip()
            data['caption'].append(cap)

assert len([cap for cap in data['caption'] if len(cap) == 0]) == 0, \
    "some caps are empty"

# data["caption"] = ["a" for _ in range(len(data['idx']))]
print(len(data['idx']))
print(len(data['caption']))

c5_df = pd.read_csv(ROOT_PATH.joinpath('data', 'processed',
                                       'cap_subsets', 'c5.csv'))

# fetch remaining data from c5
for i in data['idx']:
    if i % 1000 == 0:
        print(i)
    image_id = c5_df.at[i, 'image_id']
    image_name = c5_df.at[i, 'image_name']
    caption_id = c5_df.at[i, 'caption_id'] + "#p"
    data['image_id'].append(image_id)
    data['image_name'].append(image_name)
    data['caption_id'].append(caption_id)

# save to file
out_file = external_path.joinpath("p5.csv")
df = pd.DataFrame(data=data, columns=['image_id', 'image_name',
                                      'caption_id', 'caption'])
df.to_csv(out_file)
