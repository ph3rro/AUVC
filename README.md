This is our final project github repo for the AUVC @ MIT Beaverworks 2026. Our team name is platform 6 3/4, since our average height is 5 feet and 6 3/4 inches (and we all like Harry Potter).
We are using this code to make our AUVs autonomous, meaning we will just press one key (probably the enter key), and the AUV will complete the challenge.
The challenge consists of a 1v1 battle between 2 AUVs, and to win one AUV must flash another AUV (using its flashlights) when they are within a distance of 1 meter.
We are using a YOLO machine learning model trained on 1000+ images of these AUVs underwater to determine distance based on the bounding boxes.
In terms of movement, we use multiple nodes to get to the target depth, heading, and to tell it to go forward.
