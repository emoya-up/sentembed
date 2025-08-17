import pandas as pd

deText = pd.read_csv("data/III-001-organisierteSchuld.3pfb5.0.csv",
                     names=["de", "empty"],
                     sep=";", lineterminator="\n")
enText = pd.read_csv("data/III-004-organizedGuilt.3pw8d.0.csv",
                     names=["en"], na_filter=None)

alignedText = pd.concat([enText, deText.de], axis=1)
alignedText.replace('-', pd.NA, inplace=True)
alignedText.replace(" ", pd.NA, inplace=True)

alignedText.to_csv("data/essay1.csv")