# 1) Region Proposal 
- Sliding Window Approach for object localization is not efficiant because of large search space due to different locations and different scale of windows.
- Region Proposal methods are used in these scenarios.
- Takes an image as the input and output bounding boxes corresponding to all patches in an image that are most likely to be objects.
- Uses Image segmentaiton to identify the object containing regions.
- After Segmentation grouping of similar segments are done based on the similarities of Color ,texture size etc.
- Final number of regions that are generated are far less then Sliding window algorithm.
- Region Proposal Methods must have High Recall. In other words all the object must be covered by the regions proposed by the algorithm.
- Several region proposal methods have been proposed such as
    1) Objectness
    2) Constrained Parametric Min-Cuts for Automatic Object Segmentation
    3) Category Independent Object Proposals
    4) Randomized Prim
    5) Selective Search
- Most Commonly selective search is used. 

### Selective Search
- Selective Search is a region proposal algorithm which is fast with a very high recall. 
- It is based on computing hierarchical grouping of similar regions based on color, texture, size and shape compatibility.
- Starts with oversegmenting the image using Graph Based Segmentation algorithm propsed by [Felzenszwalb and Huttenlocher](http://cs.brown.edu/people/pfelzens/segment/) .
- Selective Search algorithm takes these oversegments as initial input and performs the following steps
    1) Add all bounding boxes corresponding to segmented parts to the list of regional proposals
    2) Group adjacent segments based on similarity
    3) Repeat 1 
- Similarity Between Regions
    1) Color Similarity
    2) Texture Similarity
    3) Size Similarity
    4) Shape Compatibility

### Refrences 
1) https://learnopencv.com/selective-search-for-object-detection-cpp-python/
2) http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf


# 2) Non Max Supression
