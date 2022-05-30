#%%
import os
import numpy as np
import xml.etree.ElementTree as ET


def change_target(xml_path, x=0, y=0):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if x == 0 and y == 0:
        while True:
            target = np.random.uniform(low=0.1, high=0.2, size=2)
            if np.linalg.norm(target) < 0.2:
                break
        x = target[0]
        y = target[1]
    else:
        target = [x, y]
    pos = str(round(target[0], 3)) + ' ' + str(round(target[1], 3)) + ' 0'
    root.find('worldbody')[-1].find('geom').attrib['pos'] = pos
    tree.write(xml_path)
    return x, y


# %%
if __name__ == '__main__':
    change_target(xml_path=os.getcwd() + '/reacher.xml')
# %%
