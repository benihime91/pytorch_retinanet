import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            bbx = member.find("bndbox")
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find("name").text

            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label,
                xmin,
                ymin,
                xmax,
                ymax,
            )
            xml_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to xml files", type=str)
    parser.add_argument(
        "--output", help="path to output csv", type=str, default="dataset.csv",
    )

    args = parser.parse_args()
    print("[INFO] Compiling csv from xml files")
    df = xml_to_csv(args.path)
    df.to_csv(args.output, index=False)
    print(f"[INFO] csv file saved to {args.output}")
