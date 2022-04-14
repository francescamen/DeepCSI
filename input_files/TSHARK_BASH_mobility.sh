#!/bin/bash
for FOLDERNAME in dataset_mobility
do
	cd "$FOLDERNAME"
	cd "MU_matrix"

	if [ ! -d 'split' ]
	then
		mkdir 'split'
	fi

	for FILENAME in ./*
	do
		echo "$FILENAME"
		FILENAMEOUTBASE=${FILENAME:2:5}

		FILENAMEOUTEND="_BE.pcapng"
		FILENAMEOUT=$"split/$FILENAMEOUTBASE$FILENAMEOUTEND"
		echo "$FILENAMEOUT"
		if [ ! -f "$FILENAMEOUT" ]
		then
			tshark -r "$FILENAME" -Y 'wlan.vht.mimo_control.feedbacktype==MU && wlan.addr==14:59:C0:5A:48:BE' -w "$FILENAMEOUT"
		fi

		FILENAMEOUTEND="_57.pcapng"
		FILENAMEOUT=$"split/$FILENAMEOUTBASE$FILENAMEOUTEND"
		echo "$FILENAMEOUT"
		if [ ! -f "$FILENAMEOUT" ]
		then
			tshark -r "$FILENAME" -Y 'wlan.vht.mimo_control.feedbacktype==MU && wlan.addr==14:59:C0:34:A2:57' -w "$FILENAMEOUT"
		fi
	done
	cd ../..
done