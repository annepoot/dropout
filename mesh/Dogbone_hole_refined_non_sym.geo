//+
SetFactory("OpenCASCADE");

Box(1) = {-5, -3, 0, 10, 6., 0.5};

//+
Cylinder(2) = {0, -6, 0, 0, 0, 1, 5, 2*Pi};
Cylinder(3) = {0, 6, -0, 0, 0, 1, 5, 2*Pi};

//+
BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Volume{3}; Delete ; }

Cylinder(2) = {0, 0, 0, 0, 0, 0.5, 0.5, 2*Pi};

BooleanDifference{ Volume{1}; Delete ; }{ Volume{2}; Delete ; }

Box(2) = {-2.5,0.,-2,5.,10,4}; 
v() = BooleanIntersection{Volume{2} ; Delete;}{ Volume{1}; } ;
BooleanFragments{ Volume{v()} ; Delete; }{ Volume{1}; Delete ;} 

Characteristic Length{ PointsOf{ Volume{:}; } } = 1.0;
Characteristic Length{ PointsOf{ Volume{v()}; } } = 0.1;
