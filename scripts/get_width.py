#!/usr/bin/env python
#
# given a histogram, find the smallest-width window that contains a
# fraction f of the events

import sys,argparse,hfile,numpy

# get the command-line arguments
parser = argparse.ArgumentParser(description='given a histogram, find the smallest-width window that contains a fraction f of the events.')
parser.add_argument('-f', type=float, help='fraction of events in the window')
parser.add_argument('-fn', type=str, help='input file name', default='')
parser.add_argument('-hist', type=str, help='name of histogram to parse')
parser.add_argument('-auto_norm', action='store_true', help='automatically normalsie the histogram to 1')
parser.add_argument('-norm', type=float, default=1.0, help='normalise the histogram by dividing by the provided value')
args = parser.parse_args()
f=args.f
if not args.fn:
    fn=sys.stdin
else:
    fn=args.fn
hname=args.hist

# read the data
a=hfile.get_array(fn, hname)
a=numpy.transpose(a)
# a=[]
# for line in sys.stdin:
#     if ((len(line)>1) and (line[0]!="#")):
#         cols=line.split()
#         if (len(a)==0):
#             for col in cols:
#                 a.append([])
#         for i in range(len(cols)):
#             a[i].append(float(cols[i]))

# normalise the histogram
if args.auto_norm:
    norm = numpy.sum(a[3]*(a[2]-a[0]))   
    a=numpy.array(a)
    if (norm>0):
        a[3] /= norm
        if (len(a)>4): a[4] /= norm
else:
    # apply the provided (optional) normalisation
    if (args.norm>0):
        a[3] /= args.norm
        if (len(a)>4): a[4] /= args.norm

# initialise the window
winsum=0.0
winmin=0.0
winmax=0.0
minwidth=1.0e100
iminlo=-1
iminhi=-1

ilo=0
ihi=0
n=len(a[0])

# loop
while True:
    # move the upper end of the window until we reach f
    while ihi<n-1 and (winsum+(a[2][ihi]-a[0][ihi])*a[3][ihi])<f:
        winsum+=(a[2][ihi]-a[0][ihi])*a[3][ihi]
        ihi+=1
    if (ihi == n-1): break

    # deal with the floating upper end
    next_add = (a[2][ihi]-a[0][ihi])*a[3][ihi]
    frac = (f-winsum)/next_add
    
    lower_end = a[0][ilo]
    upper_end = a[0][ihi]+frac*(a[2][ihi]-a[0][ihi])
    #print "  ",lower_end,upper_end,upper_end-lower_end
    if upper_end-lower_end < minwidth:
        winmin = lower_end
        winmax = upper_end
        minwidth = upper_end-lower_end
        iminlo=ilo
        iminhi=-1
        #print "  new min"

    ihi+=1
    winsum+=next_add

    # move the lower end until we're below the fraction
    while (winsum-(a[2][ilo]-a[0][ilo])*a[3][ilo])>f:
        winsum-=(a[2][ilo]-a[0][ilo])*a[3][ilo]
        ilo+=1

    # deal with the flaoting lower end
    next_sub = (a[2][ilo]-a[0][ilo])*a[3][ilo]
    frac = (winsum-f)/next_sub
    
    lower_end = a[0][ilo]+frac*(a[2][ilo]-a[0][ilo])
    upper_end = a[0][ihi]
    #print "  ",lower_end,upper_end,upper_end-lower_end
    if upper_end-lower_end < minwidth:
        winmin = lower_end
        winmax = upper_end
        minwidth = upper_end-lower_end
        iminlo=-1
        iminhi=ihi
        #print "  new min"

    ilo+=1
    winsum-=next_sub

# Now, within that window, find the median
winsum=0.0
winmid=0.0
if iminhi<0:
    ilo=iminlo
    ihi=ilo
    while (winsum+(a[2][ihi]-a[0][ihi])*a[3][ihi])<0.5*f:
        winsum+=(a[2][ihi]-a[0][ihi])*a[3][ihi]
        ihi+=1
    next_add = (a[2][ihi]-a[0][ihi])*a[3][ihi]
    frac = (0.5*f-winsum)/next_add
    winmid=a[0][ihi]+frac*(a[2][ihi]-a[0][ihi])

if iminlo<0:
    ihi=iminhi
    ilo=ihi-1
    while (winsum+(a[2][ilo]-a[0][ilo])*a[3][ilo])<0.5*f:
        winsum+=(a[2][ilo]-a[0][ilo])*a[3][ilo]
        ilo-=1
    next_add = (a[2][ilo]-a[0][ilo])*a[3][ilo]
    frac = (0.5*f-winsum)/next_add
    winmid=a[2][ilo]-frac*(a[2][ilo]-a[0][ilo])

print winmin,winmid,winmax
        
