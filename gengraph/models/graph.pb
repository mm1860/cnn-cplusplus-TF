
@
inputPlaceholder*
shape:��*
dtype0
2
labelPlaceholder*
shape:*
dtype0
[
conv1_1/truncated_normal/shapeConst*
dtype0*%
valueB"         @   
J
conv1_1/truncated_normal/meanConst*
dtype0*
valueB
 *    
L
conv1_1/truncated_normal/stddevConst*
dtype0*
valueB
 *
�#<
�
(conv1_1/truncated_normal/TruncatedNormalTruncatedNormalconv1_1/truncated_normal/shape*
dtype0*

seed *
T0*
seed2 
w
conv1_1/truncated_normal/mulMul(conv1_1/truncated_normal/TruncatedNormalconv1_1/truncated_normal/stddev*
T0
e
conv1_1/truncated_normalAddconv1_1/truncated_normal/mulconv1_1/truncated_normal/mean*
T0
k
conv1_1/weights
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
�
conv1_1/weights/AssignAssignconv1_1/weightsconv1_1/truncated_normal*
validate_shape(*"
_class
loc:@conv1_1/weights*
T0*
use_locking(
^
conv1_1/weights/readIdentityconv1_1/weights*"
_class
loc:@conv1_1/weights*
T0
�
conv1_1/Conv2DConv2Dinputconv1_1/weights/read*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
>
conv1_1/ConstConst*
valueB@*    *
dtype0
^
conv1_1/biases
VariableV2*
	container *
shape:@*
dtype0*
shared_name 
�
conv1_1/biases/AssignAssignconv1_1/biasesconv1_1/Const*!
_class
loc:@conv1_1/biases*
T0*
validate_shape(*
use_locking(
[
conv1_1/biases/readIdentityconv1_1/biases*!
_class
loc:@conv1_1/biases*
T0
_
conv1_1/BiasAddBiasAddconv1_1/Conv2Dconv1_1/biases/read*
data_formatNHWC*
T0
)
conv1_1Reluconv1_1/BiasAdd*
T0
t
pool1MaxPoolconv1_1*
ksize
*
T0*
strides
*
data_formatNHWC*
paddingSAME
[
conv2_1/truncated_normal/shapeConst*%
valueB"      @   �   *
dtype0
J
conv2_1/truncated_normal/meanConst*
valueB
 *    *
dtype0
L
conv2_1/truncated_normal/stddevConst*
valueB
 *
�#<*
dtype0
�
(conv2_1/truncated_normal/TruncatedNormalTruncatedNormalconv2_1/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 
w
conv2_1/truncated_normal/mulMul(conv2_1/truncated_normal/TruncatedNormalconv2_1/truncated_normal/stddev*
T0
e
conv2_1/truncated_normalAddconv2_1/truncated_normal/mulconv2_1/truncated_normal/mean*
T0
l
conv2_1/weights
VariableV2*
shared_name *
dtype0*
shape:@�*
	container 
�
conv2_1/weights/AssignAssignconv2_1/weightsconv2_1/truncated_normal*"
_class
loc:@conv2_1/weights*
T0*
validate_shape(*
use_locking(
^
conv2_1/weights/readIdentityconv2_1/weights*
T0*"
_class
loc:@conv2_1/weights
�
conv2_1/Conv2DConv2Dpool1conv2_1/weights/read*
use_cudnn_on_gpu(*
T0*
strides
*
data_formatNHWC*
paddingSAME
?
conv2_1/ConstConst*
dtype0*
valueB�*    
_
conv2_1/biases
VariableV2*
shape:�*
shared_name *
dtype0*
	container 
�
conv2_1/biases/AssignAssignconv2_1/biasesconv2_1/Const*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv2_1/biases
[
conv2_1/biases/readIdentityconv2_1/biases*
T0*!
_class
loc:@conv2_1/biases
_
conv2_1/BiasAddBiasAddconv2_1/Conv2Dconv2_1/biases/read*
data_formatNHWC*
T0
)
conv2_1Reluconv2_1/BiasAdd*
T0
t
pool2MaxPoolconv2_1*
strides
*
data_formatNHWC*
T0*
paddingSAME*
ksize

[
conv3_1/truncated_normal/shapeConst*
dtype0*%
valueB"      �      
J
conv3_1/truncated_normal/meanConst*
valueB
 *    *
dtype0
L
conv3_1/truncated_normal/stddevConst*
valueB
 *
�#<*
dtype0
�
(conv3_1/truncated_normal/TruncatedNormalTruncatedNormalconv3_1/truncated_normal/shape*
dtype0*

seed *
T0*
seed2 
w
conv3_1/truncated_normal/mulMul(conv3_1/truncated_normal/TruncatedNormalconv3_1/truncated_normal/stddev*
T0
e
conv3_1/truncated_normalAddconv3_1/truncated_normal/mulconv3_1/truncated_normal/mean*
T0
m
conv3_1/weights
VariableV2*
	container *
dtype0*
shared_name *
shape:��
�
conv3_1/weights/AssignAssignconv3_1/weightsconv3_1/truncated_normal*
use_locking(*
validate_shape(*
T0*"
_class
loc:@conv3_1/weights
^
conv3_1/weights/readIdentityconv3_1/weights*"
_class
loc:@conv3_1/weights*
T0
�
conv3_1/Conv2DConv2Dpool2conv3_1/weights/read*
paddingSAME*
strides
*
data_formatNHWC*
T0*
use_cudnn_on_gpu(
?
conv3_1/ConstConst*
valueB�*    *
dtype0
_
conv3_1/biases
VariableV2*
shared_name *
dtype0*
shape:�*
	container 
�
conv3_1/biases/AssignAssignconv3_1/biasesconv3_1/Const*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv3_1/biases
[
conv3_1/biases/readIdentityconv3_1/biases*
T0*!
_class
loc:@conv3_1/biases
_
conv3_1/BiasAddBiasAddconv3_1/Conv2Dconv3_1/biases/read*
T0*
data_formatNHWC
)
conv3_1Reluconv3_1/BiasAdd*
T0
t
pool3MaxPoolconv3_1*
paddingSAME*
strides
*
data_formatNHWC*
T0*
ksize

[
conv4_1/truncated_normal/shapeConst*
dtype0*%
valueB"            
J
conv4_1/truncated_normal/meanConst*
dtype0*
valueB
 *    
L
conv4_1/truncated_normal/stddevConst*
valueB
 *
�#<*
dtype0
�
(conv4_1/truncated_normal/TruncatedNormalTruncatedNormalconv4_1/truncated_normal/shape*
seed2 *
dtype0*
T0*

seed 
w
conv4_1/truncated_normal/mulMul(conv4_1/truncated_normal/TruncatedNormalconv4_1/truncated_normal/stddev*
T0
e
conv4_1/truncated_normalAddconv4_1/truncated_normal/mulconv4_1/truncated_normal/mean*
T0
m
conv4_1/weights
VariableV2*
	container *
shape:��*
dtype0*
shared_name 
�
conv4_1/weights/AssignAssignconv4_1/weightsconv4_1/truncated_normal*"
_class
loc:@conv4_1/weights*
T0*
validate_shape(*
use_locking(
^
conv4_1/weights/readIdentityconv4_1/weights*
T0*"
_class
loc:@conv4_1/weights
�
conv4_1/Conv2DConv2Dpool3conv4_1/weights/read*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0*
paddingSAME
?
conv4_1/ConstConst*
valueB�*    *
dtype0
_
conv4_1/biases
VariableV2*
	container *
shape:�*
dtype0*
shared_name 
�
conv4_1/biases/AssignAssignconv4_1/biasesconv4_1/Const*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv4_1/biases
[
conv4_1/biases/readIdentityconv4_1/biases*
T0*!
_class
loc:@conv4_1/biases
_
conv4_1/BiasAddBiasAddconv4_1/Conv2Dconv4_1/biases/read*
data_formatNHWC*
T0
)
conv4_1Reluconv4_1/BiasAdd*
T0
t
pool4MaxPoolconv4_1*
paddingSAME*
strides
*
data_formatNHWC*
T0*
ksize

b
%mentee_conv5_1/truncated_normal/shapeConst*%
valueB"            *
dtype0
Q
$mentee_conv5_1/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&mentee_conv5_1/truncated_normal/stddevConst*
valueB
 *
�#<*
dtype0
�
/mentee_conv5_1/truncated_normal/TruncatedNormalTruncatedNormal%mentee_conv5_1/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0
�
#mentee_conv5_1/truncated_normal/mulMul/mentee_conv5_1/truncated_normal/TruncatedNormal&mentee_conv5_1/truncated_normal/stddev*
T0
z
mentee_conv5_1/truncated_normalAdd#mentee_conv5_1/truncated_normal/mul$mentee_conv5_1/truncated_normal/mean*
T0
t
mentee_conv5_1/weights
VariableV2*
shared_name *
dtype0*
shape:��*
	container 
�
mentee_conv5_1/weights/AssignAssignmentee_conv5_1/weightsmentee_conv5_1/truncated_normal*)
_class
loc:@mentee_conv5_1/weights*
T0*
validate_shape(*
use_locking(
s
mentee_conv5_1/weights/readIdentitymentee_conv5_1/weights*
T0*)
_class
loc:@mentee_conv5_1/weights
�
mentee_conv5_1/Conv2DConv2Dpool4mentee_conv5_1/weights/read*
paddingSAME*
strides
*
data_formatNHWC*
T0*
use_cudnn_on_gpu(
F
mentee_conv5_1/ConstConst*
valueB�*    *
dtype0
f
mentee_conv5_1/biases
VariableV2*
	container *
shape:�*
dtype0*
shared_name 
�
mentee_conv5_1/biases/AssignAssignmentee_conv5_1/biasesmentee_conv5_1/Const*(
_class
loc:@mentee_conv5_1/biases*
T0*
validate_shape(*
use_locking(
p
mentee_conv5_1/biases/readIdentitymentee_conv5_1/biases*(
_class
loc:@mentee_conv5_1/biases*
T0
t
mentee_conv5_1/BiasAddBiasAddmentee_conv5_1/Conv2Dmentee_conv5_1/biases/read*
data_formatNHWC*
T0
7
mentee_conv5_1Relumentee_conv5_1/BiasAdd*
T0
{
pool5MaxPoolmentee_conv5_1*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC
O
fc1/truncated_normal/shapeConst*
valueB" b     *
dtype0
F
fc1/truncated_normal/meanConst*
dtype0*
valueB
 *    
H
fc1/truncated_normal/stddevConst*
dtype0*
valueB
 *
�#<
�
$fc1/truncated_normal/TruncatedNormalTruncatedNormalfc1/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0
k
fc1/truncated_normal/mulMul$fc1/truncated_normal/TruncatedNormalfc1/truncated_normal/stddev*
T0
Y
fc1/truncated_normalAddfc1/truncated_normal/mulfc1/truncated_normal/mean*
T0
b
fc1/weights
VariableV2*
shape:��� *
shared_name *
dtype0*
	container 
�
fc1/weights/AssignAssignfc1/weightsfc1/truncated_normal*
_class
loc:@fc1/weights*
T0*
validate_shape(*
use_locking(
R
fc1/weights/readIdentityfc1/weights*
_class
loc:@fc1/weights*
T0
;
	fc1/ConstConst*
valueB� *  �?*
dtype0
[

fc1/biases
VariableV2*
	container *
shape:� *
dtype0*
shared_name 
�
fc1/biases/AssignAssign
fc1/biases	fc1/Const*
_class
loc:@fc1/biases*
T0*
validate_shape(*
use_locking(
O
fc1/biases/readIdentity
fc1/biases*
_class
loc:@fc1/biases*
T0
F
fc1/Reshape/shapeConst*
dtype0*
valueB"���� b  
G
fc1/ReshapeReshapepool5fc1/Reshape/shape*
T0*
Tshape0
b

fc1/MatMulMatMulfc1/Reshapefc1/weights/read*
transpose_b( *
T0*
transpose_a( 
S
fc1/BiasAddBiasAdd
fc1/MatMulfc1/biases/read*
T0*
data_formatNHWC
&
fc1/ReluRelufc1/BiasAdd*
T0
>
dropout/keep_probConst*
dtype0*
valueB
 *   ?
B
dropout/ShapeConst*
valueB"      *
dtype0
G
dropout/random_uniform/minConst*
valueB
 *    *
dtype0
G
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0
s
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed2 *
T0*

seed *
dtype0
b
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0
l
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0
^
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0
F
dropout/addAdddropout/keep_probdropout/random_uniform*
T0
,
dropout/FloorFloordropout/add*
T0
<
dropout/divRealDivfc1/Reludropout/keep_prob*
T0
7
dropout/mulMuldropout/divdropout/Floor*
T0
.
ToInt64Castlabel*

SrcT0*

DstT0	
<
xentropy/ShapeConst*
valueB:*
dtype0
f
xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogitsdropout/mulToInt64*
T0*
Tlabels0	
3
ConstConst*
dtype0*
valueB: 
L
lossMeanxentropy/xentropyConst*
T0*

Tidx0*
	keep_dims( 
C
global_step/initial_valueConst*
value	B : *
dtype0
W
global_step
VariableV2*
	container *
shape: *
dtype0*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/initial_value*
_class
loc:@global_step*
T0*
validate_shape(*
use_locking(
R
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
valueB
 *  �?*
dtype0
A
gradients/FillFillgradients/Shapegradients/Const*
T0
O
!gradients/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
T0*
Tshape0
P
"gradients/loss_grad/Tile/multiplesConst*
valueB:*
dtype0
|
gradients/loss_grad/TileTilegradients/loss_grad/Reshape"gradients/loss_grad/Tile/multiples*

Tmultiples0*
T0
G
gradients/loss_grad/ShapeConst*
valueB:*
dtype0
D
gradients/loss_grad/Shape_1Const*
valueB *
dtype0
G
gradients/loss_grad/ConstConst*
dtype0*
valueB: 
|
gradients/loss_grad/ProdProdgradients/loss_grad/Shapegradients/loss_grad/Const*
T0*

Tidx0*
	keep_dims( 
I
gradients/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients/loss_grad/Prod_1Prodgradients/loss_grad/Shape_1gradients/loss_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
G
gradients/loss_grad/Maximum/yConst*
dtype0*
value	B :
j
gradients/loss_grad/MaximumMaximumgradients/loss_grad/Prod_1gradients/loss_grad/Maximum/y*
T0
h
gradients/loss_grad/floordivFloorDivgradients/loss_grad/Prodgradients/loss_grad/Maximum*
T0
V
gradients/loss_grad/CastCastgradients/loss_grad/floordiv*

DstT0*

SrcT0
c
gradients/loss_grad/truedivRealDivgradients/loss_grad/Tilegradients/loss_grad/Cast*
T0
?
gradients/zeros_like	ZerosLikexentropy/xentropy:1*
T0
�
0gradients/xentropy/xentropy_grad/PreventGradientPreventGradientxentropy/xentropy:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
b
/gradients/xentropy/xentropy_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������
�
+gradients/xentropy/xentropy_grad/ExpandDims
ExpandDimsgradients/loss_grad/truediv/gradients/xentropy/xentropy_grad/ExpandDims/dim*

Tdim0*
T0
�
$gradients/xentropy/xentropy_grad/mulMul+gradients/xentropy/xentropy_grad/ExpandDims0gradients/xentropy/xentropy_grad/PreventGradient*
T0
U
 gradients/dropout/mul_grad/ShapeConst*
valueB"      *
dtype0
W
"gradients/dropout/mul_grad/Shape_1Const*
valueB"      *
dtype0
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0
c
gradients/dropout/mul_grad/mulMul$gradients/xentropy/xentropy_grad/muldropout/Floor*
T0
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0
c
 gradients/dropout/mul_grad/mul_1Muldropout/div$gradients/xentropy/xentropy_grad/mul*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
U
 gradients/dropout/div_grad/ShapeConst*
valueB"      *
dtype0
K
"gradients/dropout/div_grad/Shape_1Const*
dtype0*
valueB 
�
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0
~
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependencydropout/keep_prob*
T0
�
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0
8
gradients/dropout/div_grad/NegNegfc1/Relu*
T0
k
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Negdropout/keep_prob*
T0
q
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1dropout/keep_prob*
T0
�
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0
�
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0
t
 gradients/fc1/Relu_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyfc1/Relu*
T0
w
&gradients/fc1/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/fc1/Relu_grad/ReluGrad*
data_formatNHWC*
T0

+gradients/fc1/BiasAdd_grad/tuple/group_depsNoOp!^gradients/fc1/Relu_grad/ReluGrad'^gradients/fc1/BiasAdd_grad/BiasAddGrad
�
3gradients/fc1/BiasAdd_grad/tuple/control_dependencyIdentity gradients/fc1/Relu_grad/ReluGrad,^gradients/fc1/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/fc1/Relu_grad/ReluGrad
�
5gradients/fc1/BiasAdd_grad/tuple/control_dependency_1Identity&gradients/fc1/BiasAdd_grad/BiasAddGrad,^gradients/fc1/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients/fc1/BiasAdd_grad/BiasAddGrad*
T0
�
 gradients/fc1/MatMul_grad/MatMulMatMul3gradients/fc1/BiasAdd_grad/tuple/control_dependencyfc1/weights/read*
transpose_b(*
T0*
transpose_a( 
�
"gradients/fc1/MatMul_grad/MatMul_1MatMulfc1/Reshape3gradients/fc1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
z
*gradients/fc1/MatMul_grad/tuple/group_depsNoOp!^gradients/fc1/MatMul_grad/MatMul#^gradients/fc1/MatMul_grad/MatMul_1
�
2gradients/fc1/MatMul_grad/tuple/control_dependencyIdentity gradients/fc1/MatMul_grad/MatMul+^gradients/fc1/MatMul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/fc1/MatMul_grad/MatMul
�
4gradients/fc1/MatMul_grad/tuple/control_dependency_1Identity"gradients/fc1/MatMul_grad/MatMul_1+^gradients/fc1/MatMul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/fc1/MatMul_grad/MatMul_1*
T0
]
 gradients/fc1/Reshape_grad/ShapeConst*%
valueB"            *
dtype0
�
"gradients/fc1/Reshape_grad/ReshapeReshape2gradients/fc1/MatMul_grad/tuple/control_dependency gradients/fc1/Reshape_grad/Shape*
Tshape0*
T0
�
 gradients/pool5_grad/MaxPoolGradMaxPoolGradmentee_conv5_1pool5"gradients/fc1/Reshape_grad/Reshape*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

m
&gradients/mentee_conv5_1_grad/ReluGradReluGrad gradients/pool5_grad/MaxPoolGradmentee_conv5_1*
T0
�
1gradients/mentee_conv5_1/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients/mentee_conv5_1_grad/ReluGrad*
data_formatNHWC*
T0
�
6gradients/mentee_conv5_1/BiasAdd_grad/tuple/group_depsNoOp'^gradients/mentee_conv5_1_grad/ReluGrad2^gradients/mentee_conv5_1/BiasAdd_grad/BiasAddGrad
�
>gradients/mentee_conv5_1/BiasAdd_grad/tuple/control_dependencyIdentity&gradients/mentee_conv5_1_grad/ReluGrad7^gradients/mentee_conv5_1/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients/mentee_conv5_1_grad/ReluGrad*
T0
�
@gradients/mentee_conv5_1/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/mentee_conv5_1/BiasAdd_grad/BiasAddGrad7^gradients/mentee_conv5_1/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mentee_conv5_1/BiasAdd_grad/BiasAddGrad
g
*gradients/mentee_conv5_1/Conv2D_grad/ShapeConst*%
valueB"            *
dtype0
�
8gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/mentee_conv5_1/Conv2D_grad/Shapementee_conv5_1/weights/read>gradients/mentee_conv5_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
i
,gradients/mentee_conv5_1/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0
�
9gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpool4,gradients/mentee_conv5_1/Conv2D_grad/Shape_1>gradients/mentee_conv5_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0*
paddingSAME
�
5gradients/mentee_conv5_1/Conv2D_grad/tuple/group_depsNoOp9^gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropInput:^gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropFilter
�
=gradients/mentee_conv5_1/Conv2D_grad/tuple/control_dependencyIdentity8gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropInput6^gradients/mentee_conv5_1/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropInput*
T0
�
?gradients/mentee_conv5_1/Conv2D_grad/tuple/control_dependency_1Identity9gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropFilter6^gradients/mentee_conv5_1/Conv2D_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/mentee_conv5_1/Conv2D_grad/Conv2DBackpropFilter
�
 gradients/pool4_grad/MaxPoolGradMaxPoolGradconv4_1pool4=gradients/mentee_conv5_1/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME
_
gradients/conv4_1_grad/ReluGradReluGrad gradients/pool4_grad/MaxPoolGradconv4_1*
T0
z
*gradients/conv4_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/conv4_1_grad/ReluGrad*
T0*
data_formatNHWC
�
/gradients/conv4_1/BiasAdd_grad/tuple/group_depsNoOp ^gradients/conv4_1_grad/ReluGrad+^gradients/conv4_1/BiasAdd_grad/BiasAddGrad
�
7gradients/conv4_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/conv4_1_grad/ReluGrad0^gradients/conv4_1/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv4_1_grad/ReluGrad
�
9gradients/conv4_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv4_1/BiasAdd_grad/BiasAddGrad0^gradients/conv4_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv4_1/BiasAdd_grad/BiasAddGrad
`
#gradients/conv4_1/Conv2D_grad/ShapeConst*
dtype0*%
valueB"            
�
1gradients/conv4_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv4_1/Conv2D_grad/Shapeconv4_1/weights/read7gradients/conv4_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
b
%gradients/conv4_1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"            
�
2gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpool3%gradients/conv4_1/Conv2D_grad/Shape_17gradients/conv4_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0*
paddingSAME
�
.gradients/conv4_1/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput3^gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter
�
6gradients/conv4_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv4_1/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput
�
8gradients/conv4_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv4_1/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter*
T0
�
 gradients/pool3_grad/MaxPoolGradMaxPoolGradconv3_1pool36gradients/conv4_1/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME
_
gradients/conv3_1_grad/ReluGradReluGrad gradients/pool3_grad/MaxPoolGradconv3_1*
T0
z
*gradients/conv3_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/conv3_1_grad/ReluGrad*
data_formatNHWC*
T0
�
/gradients/conv3_1/BiasAdd_grad/tuple/group_depsNoOp ^gradients/conv3_1_grad/ReluGrad+^gradients/conv3_1/BiasAdd_grad/BiasAddGrad
�
7gradients/conv3_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/conv3_1_grad/ReluGrad0^gradients/conv3_1/BiasAdd_grad/tuple/group_deps*2
_class(
&$loc:@gradients/conv3_1_grad/ReluGrad*
T0
�
9gradients/conv3_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv3_1/BiasAdd_grad/BiasAddGrad0^gradients/conv3_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv3_1/BiasAdd_grad/BiasAddGrad
`
#gradients/conv3_1/Conv2D_grad/ShapeConst*%
valueB"   8   8   �   *
dtype0
�
1gradients/conv3_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv3_1/Conv2D_grad/Shapeconv3_1/weights/read7gradients/conv3_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
b
%gradients/conv3_1/Conv2D_grad/Shape_1Const*%
valueB"      �      *
dtype0
�
2gradients/conv3_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpool2%gradients/conv3_1/Conv2D_grad/Shape_17gradients/conv3_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
strides
*
T0*
use_cudnn_on_gpu(
�
.gradients/conv3_1/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv3_1/Conv2D_grad/Conv2DBackpropInput3^gradients/conv3_1/Conv2D_grad/Conv2DBackpropFilter
�
6gradients/conv3_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv3_1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv3_1/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv3_1/Conv2D_grad/Conv2DBackpropInput
�
8gradients/conv3_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv3_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv3_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv3_1/Conv2D_grad/Conv2DBackpropFilter
�
 gradients/pool2_grad/MaxPoolGradMaxPoolGradconv2_1pool26gradients/conv3_1/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME
_
gradients/conv2_1_grad/ReluGradReluGrad gradients/pool2_grad/MaxPoolGradconv2_1*
T0
z
*gradients/conv2_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/conv2_1_grad/ReluGrad*
data_formatNHWC*
T0
�
/gradients/conv2_1/BiasAdd_grad/tuple/group_depsNoOp ^gradients/conv2_1_grad/ReluGrad+^gradients/conv2_1/BiasAdd_grad/BiasAddGrad
�
7gradients/conv2_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/conv2_1_grad/ReluGrad0^gradients/conv2_1/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/conv2_1_grad/ReluGrad
�
9gradients/conv2_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv2_1/BiasAdd_grad/BiasAddGrad0^gradients/conv2_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv2_1/BiasAdd_grad/BiasAddGrad
`
#gradients/conv2_1/Conv2D_grad/ShapeConst*%
valueB"   p   p   @   *
dtype0
�
1gradients/conv2_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2_1/Conv2D_grad/Shapeconv2_1/weights/read7gradients/conv2_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
b
%gradients/conv2_1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"      @   �   
�
2gradients/conv2_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpool1%gradients/conv2_1/Conv2D_grad/Shape_17gradients/conv2_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
.gradients/conv2_1/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv2_1/Conv2D_grad/Conv2DBackpropInput3^gradients/conv2_1/Conv2D_grad/Conv2DBackpropFilter
�
6gradients/conv2_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv2_1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv2_1/Conv2D_grad/tuple/group_deps*D
_class:
86loc:@gradients/conv2_1/Conv2D_grad/Conv2DBackpropInput*
T0
�
8gradients/conv2_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv2_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv2_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2_1/Conv2D_grad/Conv2DBackpropFilter
�
 gradients/pool1_grad/MaxPoolGradMaxPoolGradconv1_1pool16gradients/conv2_1/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

_
gradients/conv1_1_grad/ReluGradReluGrad gradients/pool1_grad/MaxPoolGradconv1_1*
T0
z
*gradients/conv1_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/conv1_1_grad/ReluGrad*
T0*
data_formatNHWC
�
/gradients/conv1_1/BiasAdd_grad/tuple/group_depsNoOp ^gradients/conv1_1_grad/ReluGrad+^gradients/conv1_1/BiasAdd_grad/BiasAddGrad
�
7gradients/conv1_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/conv1_1_grad/ReluGrad0^gradients/conv1_1/BiasAdd_grad/tuple/group_deps*2
_class(
&$loc:@gradients/conv1_1_grad/ReluGrad*
T0
�
9gradients/conv1_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv1_1/BiasAdd_grad/BiasAddGrad0^gradients/conv1_1/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/conv1_1/BiasAdd_grad/BiasAddGrad*
T0
`
#gradients/conv1_1/Conv2D_grad/ShapeConst*%
valueB"   �   �      *
dtype0
�
1gradients/conv1_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv1_1/Conv2D_grad/Shapeconv1_1/weights/read7gradients/conv1_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0*
paddingSAME
b
%gradients/conv1_1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"         @   
�
2gradients/conv1_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput%gradients/conv1_1/Conv2D_grad/Shape_17gradients/conv1_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
.gradients/conv1_1/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv1_1/Conv2D_grad/Conv2DBackpropInput3^gradients/conv1_1/Conv2D_grad/Conv2DBackpropFilter
�
6gradients/conv1_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv1_1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv1_1/Conv2D_grad/tuple/group_deps*D
_class:
86loc:@gradients/conv1_1/Conv2D_grad/Conv2DBackpropInput*
T0
�
8gradients/conv1_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv1_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv1_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv1_1/Conv2D_grad/Conv2DBackpropFilter
j
beta1_power/initial_valueConst*"
_class
loc:@conv1_1/weights*
valueB
 *fff?*
dtype0
{
beta1_power
VariableV2*
	container *
dtype0*"
_class
loc:@conv1_1/weights*
shared_name *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*"
_class
loc:@conv1_1/weights
V
beta1_power/readIdentitybeta1_power*
T0*"
_class
loc:@conv1_1/weights
j
beta2_power/initial_valueConst*"
_class
loc:@conv1_1/weights*
valueB
 *w�?*
dtype0
{
beta2_power
VariableV2*
shape: *
shared_name *"
_class
loc:@conv1_1/weights*
dtype0*
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*"
_class
loc:@conv1_1/weights*
T0*
use_locking(
V
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv1_1/weights*
T0
B
zerosConst*
dtype0*%
valueB@*    
�
conv1_1/weights/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@conv1_1/weights*
shared_name *
shape:@
�
conv1_1/weights/Adam/AssignAssignconv1_1/weights/Adamzeros*
use_locking(*
T0*"
_class
loc:@conv1_1/weights*
validate_shape(
h
conv1_1/weights/Adam/readIdentityconv1_1/weights/Adam*
T0*"
_class
loc:@conv1_1/weights
D
zeros_1Const*
dtype0*%
valueB@*    
�
conv1_1/weights/Adam_1
VariableV2*
shape:@*
shared_name *"
_class
loc:@conv1_1/weights*
dtype0*
	container 
�
conv1_1/weights/Adam_1/AssignAssignconv1_1/weights/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*"
_class
loc:@conv1_1/weights
l
conv1_1/weights/Adam_1/readIdentityconv1_1/weights/Adam_1*
T0*"
_class
loc:@conv1_1/weights
8
zeros_2Const*
valueB@*    *
dtype0
�
conv1_1/biases/Adam
VariableV2*!
_class
loc:@conv1_1/biases*
	container *
shape:@*
dtype0*
shared_name 
�
conv1_1/biases/Adam/AssignAssignconv1_1/biases/Adamzeros_2*
validate_shape(*!
_class
loc:@conv1_1/biases*
T0*
use_locking(
e
conv1_1/biases/Adam/readIdentityconv1_1/biases/Adam*
T0*!
_class
loc:@conv1_1/biases
8
zeros_3Const*
dtype0*
valueB@*    
�
conv1_1/biases/Adam_1
VariableV2*!
_class
loc:@conv1_1/biases*
	container *
shape:@*
dtype0*
shared_name 
�
conv1_1/biases/Adam_1/AssignAssignconv1_1/biases/Adam_1zeros_3*!
_class
loc:@conv1_1/biases*
T0*
validate_shape(*
use_locking(
i
conv1_1/biases/Adam_1/readIdentityconv1_1/biases/Adam_1*!
_class
loc:@conv1_1/biases*
T0
E
zeros_4Const*&
valueB@�*    *
dtype0
�
conv2_1/weights/Adam
VariableV2*
shape:@�*
shared_name *"
_class
loc:@conv2_1/weights*
dtype0*
	container 
�
conv2_1/weights/Adam/AssignAssignconv2_1/weights/Adamzeros_4*
validate_shape(*"
_class
loc:@conv2_1/weights*
T0*
use_locking(
h
conv2_1/weights/Adam/readIdentityconv2_1/weights/Adam*"
_class
loc:@conv2_1/weights*
T0
E
zeros_5Const*&
valueB@�*    *
dtype0
�
conv2_1/weights/Adam_1
VariableV2*"
_class
loc:@conv2_1/weights*
	container *
shape:@�*
dtype0*
shared_name 
�
conv2_1/weights/Adam_1/AssignAssignconv2_1/weights/Adam_1zeros_5*
use_locking(*
T0*"
_class
loc:@conv2_1/weights*
validate_shape(
l
conv2_1/weights/Adam_1/readIdentityconv2_1/weights/Adam_1*"
_class
loc:@conv2_1/weights*
T0
9
zeros_6Const*
valueB�*    *
dtype0
�
conv2_1/biases/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@conv2_1/biases*
shared_name *
shape:�
�
conv2_1/biases/Adam/AssignAssignconv2_1/biases/Adamzeros_6*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv2_1/biases
e
conv2_1/biases/Adam/readIdentityconv2_1/biases/Adam*!
_class
loc:@conv2_1/biases*
T0
9
zeros_7Const*
valueB�*    *
dtype0
�
conv2_1/biases/Adam_1
VariableV2*
shape:�*
shared_name *!
_class
loc:@conv2_1/biases*
dtype0*
	container 
�
conv2_1/biases/Adam_1/AssignAssignconv2_1/biases/Adam_1zeros_7*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv2_1/biases
i
conv2_1/biases/Adam_1/readIdentityconv2_1/biases/Adam_1*
T0*!
_class
loc:@conv2_1/biases
F
zeros_8Const*
dtype0*'
valueB��*    
�
conv3_1/weights/Adam
VariableV2*
	container *
dtype0*"
_class
loc:@conv3_1/weights*
shared_name *
shape:��
�
conv3_1/weights/Adam/AssignAssignconv3_1/weights/Adamzeros_8*
validate_shape(*"
_class
loc:@conv3_1/weights*
T0*
use_locking(
h
conv3_1/weights/Adam/readIdentityconv3_1/weights/Adam*"
_class
loc:@conv3_1/weights*
T0
F
zeros_9Const*'
valueB��*    *
dtype0
�
conv3_1/weights/Adam_1
VariableV2*
shape:��*
shared_name *"
_class
loc:@conv3_1/weights*
dtype0*
	container 
�
conv3_1/weights/Adam_1/AssignAssignconv3_1/weights/Adam_1zeros_9*
use_locking(*
validate_shape(*
T0*"
_class
loc:@conv3_1/weights
l
conv3_1/weights/Adam_1/readIdentityconv3_1/weights/Adam_1*
T0*"
_class
loc:@conv3_1/weights
:
zeros_10Const*
dtype0*
valueB�*    
�
conv3_1/biases/Adam
VariableV2*!
_class
loc:@conv3_1/biases*
	container *
shape:�*
dtype0*
shared_name 
�
conv3_1/biases/Adam/AssignAssignconv3_1/biases/Adamzeros_10*
use_locking(*
T0*!
_class
loc:@conv3_1/biases*
validate_shape(
e
conv3_1/biases/Adam/readIdentityconv3_1/biases/Adam*!
_class
loc:@conv3_1/biases*
T0
:
zeros_11Const*
dtype0*
valueB�*    
�
conv3_1/biases/Adam_1
VariableV2*!
_class
loc:@conv3_1/biases*
	container *
shape:�*
dtype0*
shared_name 
�
conv3_1/biases/Adam_1/AssignAssignconv3_1/biases/Adam_1zeros_11*!
_class
loc:@conv3_1/biases*
T0*
validate_shape(*
use_locking(
i
conv3_1/biases/Adam_1/readIdentityconv3_1/biases/Adam_1*
T0*!
_class
loc:@conv3_1/biases
G
zeros_12Const*'
valueB��*    *
dtype0
�
conv4_1/weights/Adam
VariableV2*"
_class
loc:@conv4_1/weights*
	container *
shape:��*
dtype0*
shared_name 
�
conv4_1/weights/Adam/AssignAssignconv4_1/weights/Adamzeros_12*"
_class
loc:@conv4_1/weights*
T0*
validate_shape(*
use_locking(
h
conv4_1/weights/Adam/readIdentityconv4_1/weights/Adam*
T0*"
_class
loc:@conv4_1/weights
G
zeros_13Const*'
valueB��*    *
dtype0
�
conv4_1/weights/Adam_1
VariableV2*"
_class
loc:@conv4_1/weights*
	container *
shape:��*
dtype0*
shared_name 
�
conv4_1/weights/Adam_1/AssignAssignconv4_1/weights/Adam_1zeros_13*
validate_shape(*"
_class
loc:@conv4_1/weights*
T0*
use_locking(
l
conv4_1/weights/Adam_1/readIdentityconv4_1/weights/Adam_1*"
_class
loc:@conv4_1/weights*
T0
:
zeros_14Const*
valueB�*    *
dtype0
�
conv4_1/biases/Adam
VariableV2*
shape:�*
shared_name *!
_class
loc:@conv4_1/biases*
dtype0*
	container 
�
conv4_1/biases/Adam/AssignAssignconv4_1/biases/Adamzeros_14*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv4_1/biases
e
conv4_1/biases/Adam/readIdentityconv4_1/biases/Adam*!
_class
loc:@conv4_1/biases*
T0
:
zeros_15Const*
valueB�*    *
dtype0
�
conv4_1/biases/Adam_1
VariableV2*
shape:�*
shared_name *!
_class
loc:@conv4_1/biases*
dtype0*
	container 
�
conv4_1/biases/Adam_1/AssignAssignconv4_1/biases/Adam_1zeros_15*
use_locking(*
validate_shape(*
T0*!
_class
loc:@conv4_1/biases
i
conv4_1/biases/Adam_1/readIdentityconv4_1/biases/Adam_1*!
_class
loc:@conv4_1/biases*
T0
G
zeros_16Const*'
valueB��*    *
dtype0
�
mentee_conv5_1/weights/Adam
VariableV2*
shape:��*
shared_name *)
_class
loc:@mentee_conv5_1/weights*
dtype0*
	container 
�
"mentee_conv5_1/weights/Adam/AssignAssignmentee_conv5_1/weights/Adamzeros_16*
use_locking(*
T0*)
_class
loc:@mentee_conv5_1/weights*
validate_shape(
}
 mentee_conv5_1/weights/Adam/readIdentitymentee_conv5_1/weights/Adam*)
_class
loc:@mentee_conv5_1/weights*
T0
G
zeros_17Const*'
valueB��*    *
dtype0
�
mentee_conv5_1/weights/Adam_1
VariableV2*
shared_name *
dtype0*
shape:��*
	container *)
_class
loc:@mentee_conv5_1/weights
�
$mentee_conv5_1/weights/Adam_1/AssignAssignmentee_conv5_1/weights/Adam_1zeros_17*
validate_shape(*)
_class
loc:@mentee_conv5_1/weights*
T0*
use_locking(
�
"mentee_conv5_1/weights/Adam_1/readIdentitymentee_conv5_1/weights/Adam_1*
T0*)
_class
loc:@mentee_conv5_1/weights
:
zeros_18Const*
dtype0*
valueB�*    
�
mentee_conv5_1/biases/Adam
VariableV2*(
_class
loc:@mentee_conv5_1/biases*
	container *
shape:�*
dtype0*
shared_name 
�
!mentee_conv5_1/biases/Adam/AssignAssignmentee_conv5_1/biases/Adamzeros_18*(
_class
loc:@mentee_conv5_1/biases*
T0*
validate_shape(*
use_locking(
z
mentee_conv5_1/biases/Adam/readIdentitymentee_conv5_1/biases/Adam*(
_class
loc:@mentee_conv5_1/biases*
T0
:
zeros_19Const*
dtype0*
valueB�*    
�
mentee_conv5_1/biases/Adam_1
VariableV2*
shared_name *
dtype0*
shape:�*
	container *(
_class
loc:@mentee_conv5_1/biases
�
#mentee_conv5_1/biases/Adam_1/AssignAssignmentee_conv5_1/biases/Adam_1zeros_19*
validate_shape(*(
_class
loc:@mentee_conv5_1/biases*
T0*
use_locking(
~
!mentee_conv5_1/biases/Adam_1/readIdentitymentee_conv5_1/biases/Adam_1*(
_class
loc:@mentee_conv5_1/biases*
T0
@
zeros_20Const*
dtype0* 
valueB��� *    
�
fc1/weights/Adam
VariableV2*
shared_name *
dtype0*
shape:��� *
	container *
_class
loc:@fc1/weights
�
fc1/weights/Adam/AssignAssignfc1/weights/Adamzeros_20*
use_locking(*
T0*
_class
loc:@fc1/weights*
validate_shape(
\
fc1/weights/Adam/readIdentityfc1/weights/Adam*
_class
loc:@fc1/weights*
T0
@
zeros_21Const* 
valueB��� *    *
dtype0
�
fc1/weights/Adam_1
VariableV2*
shape:��� *
shared_name *
_class
loc:@fc1/weights*
dtype0*
	container 
�
fc1/weights/Adam_1/AssignAssignfc1/weights/Adam_1zeros_21*
use_locking(*
T0*
_class
loc:@fc1/weights*
validate_shape(
`
fc1/weights/Adam_1/readIdentityfc1/weights/Adam_1*
T0*
_class
loc:@fc1/weights
:
zeros_22Const*
dtype0*
valueB� *    

fc1/biases/Adam
VariableV2*
shared_name *
dtype0*
shape:� *
	container *
_class
loc:@fc1/biases
�
fc1/biases/Adam/AssignAssignfc1/biases/Adamzeros_22*
use_locking(*
T0*
_class
loc:@fc1/biases*
validate_shape(
Y
fc1/biases/Adam/readIdentityfc1/biases/Adam*
_class
loc:@fc1/biases*
T0
:
zeros_23Const*
dtype0*
valueB� *    
�
fc1/biases/Adam_1
VariableV2*
shape:� *
shared_name *
_class
loc:@fc1/biases*
dtype0*
	container 
�
fc1/biases/Adam_1/AssignAssignfc1/biases/Adam_1zeros_23*
use_locking(*
T0*
_class
loc:@fc1/biases*
validate_shape(
]
fc1/biases/Adam_1/readIdentityfc1/biases/Adam_1*
_class
loc:@fc1/biases*
T0
@
train/learning_rateConst*
valueB
 *��8*
dtype0
8
train/beta1Const*
valueB
 *fff?*
dtype0
8
train/beta2Const*
dtype0*
valueB
 *w�?
:
train/epsilonConst*
valueB
 *w�+2*
dtype0
�
&train/update_conv1_1/weights/ApplyAdam	ApplyAdamconv1_1/weightsconv1_1/weights/Adamconv1_1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/conv1_1/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv1_1/weights*
T0*
use_locking( 
�
%train/update_conv1_1/biases/ApplyAdam	ApplyAdamconv1_1/biasesconv1_1/biases/Adamconv1_1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/conv1_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@conv1_1/biases
�
&train/update_conv2_1/weights/ApplyAdam	ApplyAdamconv2_1/weightsconv2_1/weights/Adamconv2_1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/conv2_1/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2_1/weights*
T0*
use_locking( 
�
%train/update_conv2_1/biases/ApplyAdam	ApplyAdamconv2_1/biasesconv2_1/biases/Adamconv2_1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/conv2_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@conv2_1/biases
�
&train/update_conv3_1/weights/ApplyAdam	ApplyAdamconv3_1/weightsconv3_1/weights/Adamconv3_1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/conv3_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv3_1/weights
�
%train/update_conv3_1/biases/ApplyAdam	ApplyAdamconv3_1/biasesconv3_1/biases/Adamconv3_1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/conv3_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@conv3_1/biases
�
&train/update_conv4_1/weights/ApplyAdam	ApplyAdamconv4_1/weightsconv4_1/weights/Adamconv4_1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/conv4_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv4_1/weights
�
%train/update_conv4_1/biases/ApplyAdam	ApplyAdamconv4_1/biasesconv4_1/biases/Adamconv4_1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/conv4_1/BiasAdd_grad/tuple/control_dependency_1*!
_class
loc:@conv4_1/biases*
T0*
use_locking( 
�
-train/update_mentee_conv5_1/weights/ApplyAdam	ApplyAdammentee_conv5_1/weightsmentee_conv5_1/weights/Adammentee_conv5_1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon?gradients/mentee_conv5_1/Conv2D_grad/tuple/control_dependency_1*)
_class
loc:@mentee_conv5_1/weights*
T0*
use_locking( 
�
,train/update_mentee_conv5_1/biases/ApplyAdam	ApplyAdammentee_conv5_1/biasesmentee_conv5_1/biases/Adammentee_conv5_1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon@gradients/mentee_conv5_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@mentee_conv5_1/biases
�
"train/update_fc1/weights/ApplyAdam	ApplyAdamfc1/weightsfc1/weights/Adamfc1/weights/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon4gradients/fc1/MatMul_grad/tuple/control_dependency_1*
_class
loc:@fc1/weights*
T0*
use_locking( 
�
!train/update_fc1/biases/ApplyAdam	ApplyAdam
fc1/biasesfc1/biases/Adamfc1/biases/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon5gradients/fc1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc1/biases
�
	train/mulMulbeta1_power/readtrain/beta1'^train/update_conv1_1/weights/ApplyAdam&^train/update_conv1_1/biases/ApplyAdam'^train/update_conv2_1/weights/ApplyAdam&^train/update_conv2_1/biases/ApplyAdam'^train/update_conv3_1/weights/ApplyAdam&^train/update_conv3_1/biases/ApplyAdam'^train/update_conv4_1/weights/ApplyAdam&^train/update_conv4_1/biases/ApplyAdam.^train/update_mentee_conv5_1/weights/ApplyAdam-^train/update_mentee_conv5_1/biases/ApplyAdam#^train/update_fc1/weights/ApplyAdam"^train/update_fc1/biases/ApplyAdam*"
_class
loc:@conv1_1/weights*
T0
�
train/AssignAssignbeta1_power	train/mul*
validate_shape(*"
_class
loc:@conv1_1/weights*
T0*
use_locking( 
�
train/mul_1Mulbeta2_power/readtrain/beta2'^train/update_conv1_1/weights/ApplyAdam&^train/update_conv1_1/biases/ApplyAdam'^train/update_conv2_1/weights/ApplyAdam&^train/update_conv2_1/biases/ApplyAdam'^train/update_conv3_1/weights/ApplyAdam&^train/update_conv3_1/biases/ApplyAdam'^train/update_conv4_1/weights/ApplyAdam&^train/update_conv4_1/biases/ApplyAdam.^train/update_mentee_conv5_1/weights/ApplyAdam-^train/update_mentee_conv5_1/biases/ApplyAdam#^train/update_fc1/weights/ApplyAdam"^train/update_fc1/biases/ApplyAdam*
T0*"
_class
loc:@conv1_1/weights
�
train/Assign_1Assignbeta2_powertrain/mul_1*
validate_shape(*"
_class
loc:@conv1_1/weights*
T0*
use_locking( 
�
train/updateNoOp'^train/update_conv1_1/weights/ApplyAdam&^train/update_conv1_1/biases/ApplyAdam'^train/update_conv2_1/weights/ApplyAdam&^train/update_conv2_1/biases/ApplyAdam'^train/update_conv3_1/weights/ApplyAdam&^train/update_conv3_1/biases/ApplyAdam'^train/update_conv4_1/weights/ApplyAdam&^train/update_conv4_1/biases/ApplyAdam.^train/update_mentee_conv5_1/weights/ApplyAdam-^train/update_mentee_conv5_1/biases/ApplyAdam#^train/update_fc1/weights/ApplyAdam"^train/update_fc1/biases/ApplyAdam^train/Assign^train/Assign_1
d
train/valueConst^train/update*
_class
loc:@global_step*
value	B :*
dtype0
h
train	AssignAddglobal_steptrain/value*
use_locking( *
T0*
_class
loc:@global_step
�
init_all_vars_opNoOp^conv1_1/weights/Assign^conv1_1/biases/Assign^conv2_1/weights/Assign^conv2_1/biases/Assign^conv3_1/weights/Assign^conv3_1/biases/Assign^conv4_1/weights/Assign^conv4_1/biases/Assign^mentee_conv5_1/weights/Assign^mentee_conv5_1/biases/Assign^fc1/weights/Assign^fc1/biases/Assign^global_step/Assign^beta1_power/Assign^beta2_power/Assign^conv1_1/weights/Adam/Assign^conv1_1/weights/Adam_1/Assign^conv1_1/biases/Adam/Assign^conv1_1/biases/Adam_1/Assign^conv2_1/weights/Adam/Assign^conv2_1/weights/Adam_1/Assign^conv2_1/biases/Adam/Assign^conv2_1/biases/Adam_1/Assign^conv3_1/weights/Adam/Assign^conv3_1/weights/Adam_1/Assign^conv3_1/biases/Adam/Assign^conv3_1/biases/Adam_1/Assign^conv4_1/weights/Adam/Assign^conv4_1/weights/Adam_1/Assign^conv4_1/biases/Adam/Assign^conv4_1/biases/Adam_1/Assign#^mentee_conv5_1/weights/Adam/Assign%^mentee_conv5_1/weights/Adam_1/Assign"^mentee_conv5_1/biases/Adam/Assign$^mentee_conv5_1/biases/Adam_1/Assign^fc1/weights/Adam/Assign^fc1/weights/Adam_1/Assign^fc1/biases/Adam/Assign^fc1/biases/Adam_1/Assign"