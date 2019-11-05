#!/opt/twitter/bin/python

import sys
import string
import random

# read in the lines
f = open("lines.txt", "r")
o = open("triplets.txt", "w")

message = "APPLYAPPROPRIATECOLORCHECKER"

i = 0
for line in f:
	line = line.upper().strip();
	line = line.translate(None, string.punctuation)
	print line
	line = "".join(line.split())
	print line

	#randomly add in a letter
	rand_index = random.randint(1, len(line));
	line = line[:rand_index] + message[i] + line[rand_index:]

	i = i + 1
	print line

	# split into triplets
	trips = [line[j:j+3] for j in xrange(0, len(line), 3)]
	trips.sort()
	output = " ".join(trips)
	print output
	o.write(output + "\n")



