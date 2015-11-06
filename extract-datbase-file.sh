#!/bin/bash

echo "Exporting database info from Banshee..."
sqlite3 ~/.config/banshee-1/banshee.db .dump CoreTracks | grep INSERT | sed "s/INSERT INTO \"CoreTracks\" VALUES(//;s/);//" > banshee-tracks

echo "Finished exporting database info from Banshee. Created file banshee-tracks.s"

