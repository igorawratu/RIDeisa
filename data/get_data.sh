#!/bin/bash
wget https://cloud.oca.eu/index.php/s/8ASLH42i5DZ48ow/download
unzip download
mv RIDeisa\ datasets/* .
unzip jackknife_test.zip
unzip sep_test.zip

rm -r RIDeisa\ datasets
rm download
rm jackknife_test.zip
rm sep_test.zip