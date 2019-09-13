# Hough-Transform
The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing.[1] The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.

This is the hough transform implementation from scratch for detecting lines and circles in any given image. 

## Objective:

1. To detect the red and blue lines on the image.
2. To detect all the coins in the given image.

## Approach:

### Lines:

For detecting the lines, we are using the Hough transform. The basic idea is to let each pixel "vote".
We collect these votes in the accumulator cells.

In our case, the accumulator is a 2D array. Different θ values is in the horizontal axis and p values are
on the vertical axis.

Next, I am using Sobel to detect the edges on the image. Then I loop through every pixel of the edge
detected image. There are two possibilities, either the pixel value is zero or a non-zero value.

If we detect a pixel with zero value, we ignore that pixel as it is not an edge and so it can’t be a line.

If we detect a pixel with non-zero value, then we calculate p for all the θ = -90 to +90 and voting in the
corresponding accumulator cell (θ, p). Each pixel value will generate a sinusoidal curve.

Now we find the points in the accumulator where there are lot of votes. These points are the
parameter for the lines in the original image.

### Circles:

For detecting circles, we follow the similar approach as lines. However, instead of finding (θ, p), we try
to find the triplet (a,b,R) where (a,b) are the coordinates of the center and R is the radius of the circle.
In this we create a 3-D parameter space. We start similar as lines and collect votes for every non-zero
pixels. For a given R, we find a and b as follow:
 
 a = x1 - Rcosθ 
 b = y1 - Rsinθ 

θ ranges from 0 to 360 degrees and collect votes in ab space. We repeat the process for different R
values. The points where there are lot of votes represent the center and Radius of the circle.

## Results:

a) Total red lines detected = 6.

To detect the red lines, I try to find the points which are greater than certain threshold. Further
filtration was required as I was getting certain blue lines even after applying the threshold. As
the red lines are straight, I filtered out them on the basis of angles. From the threshold points,
I find out the point which are at angles near zero. I know for sure that these would be red
lines. There was one more challenge in it. I was getting a bundle of lines on each red line. For
this, I clustered all the lines with similar p values and take the middle line out of each cluster.

Using the above approach, I was able to detect 6 lines. The result is shown in output folder.

b) Total blue lines detected = 8.

The approach used for this is similar as red lines. However, the blue lines were at -36-
degree level. Hence, I filtered out the blue lines by taking out the lines only near my
desired angle.

I was able to detect 8 line out of 9. The reason for not detecting the smaller line is that
as the line is so small, there were not enough votes so that it crosses the threshold.
When I tried to lower the threshold, I was getting many undesired lines which were
not even in the image. Hence, I kept the threshold above a certain level.

c) Total coins detected = 17.

As the coins are not of same radius, I tried to find out the circles with radius ranging
from 21 to 23. Then, I found out the points which contains votes above a certain
threshold. There were many concentric circles coming out near each coin, i.e. near
(a, b, R) point. To filter out the other concentric circles, I was just considering only
one circles near each (a, b, R) and ignoring all the nearby points.

Using the above approach I was able to detect all the 17 coins on the image.

## References:

[1] https://alyssaq.github.io/2014/understanding-hough-transform/

[2]https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html

[3] https://github.com/SunilVasu/Circle-Hough-Transform/blob/master/HoughTransform_FOR_GIVEN_TEST_IMAGE.py

[4] http://aishack.in/tutorials/hough-transform-normal/

[5] http://www.aishack.in/tutorials/circle-hough-transform/
