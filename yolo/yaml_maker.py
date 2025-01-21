import yaml 

data = {
    "train" : f'data/tmp/train/',
        "val" : f'data/tmp/valid/',
        "test" : f'data/tmp/test/',
        "names" : {0 : 'tomato', 1: 'gripper'}}

with open('./tmp.yaml', 'w') as f :
    yaml.dump(data, f)

# check written file
with open('./tmp.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)