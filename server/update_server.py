import os
import pathlib


data_dir = "/home/ad/PycharmProjects/Sound_processing/venv/m"
data_dir = pathlib.Path(data_dir)


def scp(host, item_: pathlib.Path):
    if 'aime' in host.lower():
        des_dir = '/mnt/disk2/minh'
    else:
        des_dir = "/mnt/disk1/minh/2021"
    command = 'echo Nothing happens'
    if item_.is_file():
        command = f'scp -p {item_.__str__()} {host}:{des_dir}'
    elif item_.is_dir():
        command = f'scp -rp {item_.__str__()} {host}:{des_dir}'
    os.system(command)


for item in data_dir.iterdir():
    if item.name != 'data' and item.name != 'config' and item.name != 'server'\
            and item.name != 'Track' and ('.git' not in item.name):
        print(item.name)
        scp(host='aimenext', item_=item)
        scp(host='aiotlab', item_=item)


