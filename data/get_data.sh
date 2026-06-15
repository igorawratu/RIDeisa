#!/bin/bash
wget https://cloud.oca.eu/index.php/s/8ASLH42i5DZ48ow/download
unzip download
mv RIDeisa\ datasets/* .
unzip galfield.zip

rm -r RIDeisa\ datasets
rm download
rm galfield.zip