* Technical Analysis Golden/Death Cross Detector

Objective: According to some TA professionals, when the 200 day moving average crosses the 50 day moving average, 
this could be related to a trend reversal (bull/bear market).

This script uses Pandas to get Yahoo BTC daily price data, so it can calculate the 200/50 day moving average, 
then uses last 20 day prices to estimate if they are intercepting or not.

Math was used to detect the interception. numpy, pycse and uncertainties were used to achieve the result.

The script can plot the 200/50MA chart also the death/golden cross if it is happening.

Credits to the [Kitchin Research Group](http://kitchingroup.cheme.cmu.edu/blog/2013/07/04/Estimating-where-two-functions-intersect-using-data/)