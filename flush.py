import os
import glob
import sh

files = glob.glob('/Users/anubhavjain/Desktop/NLP_Pdf_extract/paper1/pdf/*')
for f in files:
    sh.rm(f)

files = glob.glob('/Users/anubhavjain/Desktop/NLP_Pdf_extract/paper1/docx/*')
for f in files:
    sh.rm(f)

files = glob.glob('/Users/anubhavjain/Desktop/NLP_Pdf_extract/paper1/html/*')
for f in files:
    sh.rm(f)