import xml.etree.ElementTree as ET
import random

vTypesFile = 'carlavtypes.rou.xml'
vTypes = []
elementTreeVT = ET.parse(vTypesFile)
root = elementTreeVT.getroot()
for child in root:
    if child.tag == 'vType':
        vtid = child.attrib['id']
        vTypes.append(vtid)

routeFile = 'output.rou.xml'
elementTreeR = ET.parse(routeFile)
root = elementTreeR.getroot()
for child in root:
    if child.tag == 'vehicle':
        child.set('type', random.choice(vTypes))

elementTreeR.write('Town06.rou.xml', encoding="utf-8", xml_declaration=True)
