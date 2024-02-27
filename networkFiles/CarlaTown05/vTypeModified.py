import xml.etree.ElementTree as ET
from typing import List
import random


def get_all_vTypes(vTypesFile: str):
    vTypes = []
    elementTreeVT = ET.parse(vTypesFile)
    root = elementTreeVT.getroot()
    for child in root:
        if child.tag == 'vType':
            vtid = child.attrib['id']
            vTypes.append(vtid)
    return vTypes

def rewrite_route_file(
    originalFile: str,
    vTypes: List[str], 
    rewriteFile: str
):
    elementTreeR = ET.parse(originalFile)
    root = elementTreeR.getroot()
    for child in root:
        if child.tag == 'vehicle':
            child.set('type', random.choice(vTypes))

    elementTreeR.write(
        rewriteFile, 
        encoding="utf-8", 
        xml_declaration=True
    )

if __name__ == '__main__':
    originalFile = './output.rou.xml'
    vTypesFile = 'carlavtypes.rou.xml'
    rewriteFile = 'Town05.rou.xml'
    vTypes = get_all_vTypes(vTypesFile)
    rewrite_route_file(originalFile, vTypes, rewriteFile)
