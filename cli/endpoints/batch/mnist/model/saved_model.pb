̋
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ApplyGradientDescent
var"T?

alpha"T

delta"T
out"T?" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
W
InTopKV2
predictions
targets"T
k"T
	precision
"
Ttype0:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'??
n
	network/XPlaceholder*
dtype0*(
_output_shapes
:??????????*
shape:??????????
N
	network/yPlaceholder*
shape:*
dtype0	*
_output_shapes
:
?
*h1/kernel/Initializer/random_uniform/shapeConst*
valueB"  ,  *
_class
loc:@h1/kernel*
dtype0*
_output_shapes
:
?
(h1/kernel/Initializer/random_uniform/minConst*
valueB
 *?]??*
_class
loc:@h1/kernel*
dtype0*
_output_shapes
: 
?
(h1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?]?=*
_class
loc:@h1/kernel
?
2h1/kernel/Initializer/random_uniform/RandomUniformRandomUniform*h1/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@h1/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
(h1/kernel/Initializer/random_uniform/subSub(h1/kernel/Initializer/random_uniform/max(h1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@h1/kernel*
_output_shapes
: 
?
(h1/kernel/Initializer/random_uniform/mulMul2h1/kernel/Initializer/random_uniform/RandomUniform(h1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
??*
T0*
_class
loc:@h1/kernel
?
$h1/kernel/Initializer/random_uniformAdd(h1/kernel/Initializer/random_uniform/mul(h1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@h1/kernel* 
_output_shapes
:
??
?
	h1/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *
_class
loc:@h1/kernel*
	container *
shape:
??
?
h1/kernel/AssignAssign	h1/kernel$h1/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@h1/kernel*
validate_shape(* 
_output_shapes
:
??
n
h1/kernel/readIdentity	h1/kernel*
T0*
_class
loc:@h1/kernel* 
_output_shapes
:
??
?
h1/bias/Initializer/zerosConst*
valueB?*    *
_class
loc:@h1/bias*
dtype0*
_output_shapes	
:?
?
h1/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *
_class
loc:@h1/bias*
	container 
?
h1/bias/AssignAssignh1/biash1/bias/Initializer/zeros*
_class
loc:@h1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
c
h1/bias/readIdentityh1/bias*
_output_shapes	
:?*
T0*
_class
loc:@h1/bias
?
network/h1/MatMulMatMul	network/Xh1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:??????????*
transpose_a( 
?
network/h1/BiasAddBiasAddnetwork/h1/MatMulh1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:??????????
^
network/h1/ReluRelunetwork/h1/BiasAdd*(
_output_shapes
:??????????*
T0
?
*h2/kernel/Initializer/random_uniform/shapeConst*
valueB",  d   *
_class
loc:@h2/kernel*
dtype0*
_output_shapes
:
?
(h2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *????*
_class
loc:@h2/kernel
?
(h2/kernel/Initializer/random_uniform/maxConst*
valueB
 *???=*
_class
loc:@h2/kernel*
dtype0*
_output_shapes
: 
?
2h2/kernel/Initializer/random_uniform/RandomUniformRandomUniform*h2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	?d*

seed *
T0*
_class
loc:@h2/kernel*
seed2 *
dtype0
?
(h2/kernel/Initializer/random_uniform/subSub(h2/kernel/Initializer/random_uniform/max(h2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@h2/kernel*
_output_shapes
: 
?
(h2/kernel/Initializer/random_uniform/mulMul2h2/kernel/Initializer/random_uniform/RandomUniform(h2/kernel/Initializer/random_uniform/sub*
_class
loc:@h2/kernel*
_output_shapes
:	?d*
T0
?
$h2/kernel/Initializer/random_uniformAdd(h2/kernel/Initializer/random_uniform/mul(h2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@h2/kernel*
_output_shapes
:	?d
?
	h2/kernel
VariableV2*
	container *
shape:	?d*
dtype0*
_output_shapes
:	?d*
shared_name *
_class
loc:@h2/kernel
?
h2/kernel/AssignAssign	h2/kernel$h2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@h2/kernel*
validate_shape(*
_output_shapes
:	?d
m
h2/kernel/readIdentity	h2/kernel*
_class
loc:@h2/kernel*
_output_shapes
:	?d*
T0
?
h2/bias/Initializer/zerosConst*
valueBd*    *
_class
loc:@h2/bias*
dtype0*
_output_shapes
:d
?
h2/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *
_class
loc:@h2/bias*
	container 
?
h2/bias/AssignAssignh2/biash2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@h2/bias*
validate_shape(*
_output_shapes
:d
b
h2/bias/readIdentityh2/bias*
T0*
_class
loc:@h2/bias*
_output_shapes
:d
?
network/h2/MatMulMatMulnetwork/h1/Reluh2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:?????????d*
transpose_a( 
?
network/h2/BiasAddBiasAddnetwork/h2/MatMulh2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????d
]
network/h2/ReluRelunetwork/h2/BiasAdd*'
_output_shapes
:?????????d*
T0
?
.output/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"d   
   * 
_class
loc:@output/kernel*
dtype0
?
,output/kernel/Initializer/random_uniform/minConst*
valueB
 *?'o?* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 
?
,output/kernel/Initializer/random_uniform/maxConst*
valueB
 *?'o>* 
_class
loc:@output/kernel*
dtype0*
_output_shapes
: 
?
6output/kernel/Initializer/random_uniform/RandomUniformRandomUniform.output/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:d
*

seed *
T0* 
_class
loc:@output/kernel
?
,output/kernel/Initializer/random_uniform/subSub,output/kernel/Initializer/random_uniform/max,output/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@output/kernel*
_output_shapes
: 
?
,output/kernel/Initializer/random_uniform/mulMul6output/kernel/Initializer/random_uniform/RandomUniform,output/kernel/Initializer/random_uniform/sub*
_output_shapes

:d
*
T0* 
_class
loc:@output/kernel
?
(output/kernel/Initializer/random_uniformAdd,output/kernel/Initializer/random_uniform/mul,output/kernel/Initializer/random_uniform/min*
_output_shapes

:d
*
T0* 
_class
loc:@output/kernel
?
output/kernel
VariableV2* 
_class
loc:@output/kernel*
	container *
shape
:d
*
dtype0*
_output_shapes

:d
*
shared_name 
?
output/kernel/AssignAssignoutput/kernel(output/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes

:d

x
output/kernel/readIdentityoutput/kernel*
T0* 
_class
loc:@output/kernel*
_output_shapes

:d

?
output/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@output/bias*
dtype0*
_output_shapes
:

?
output/bias
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@output/bias*
	container 
?
output/bias/AssignAssignoutput/biasoutput/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:

n
output/bias/readIdentityoutput/bias*
T0*
_class
loc:@output/bias*
_output_shapes
:

?
network/output/MatMulMatMulnetwork/h2/Reluoutput/kernel/read*
T0*'
_output_shapes
:?????????
*
transpose_a( *
transpose_b( 
?
network/output/BiasAddBiasAddnetwork/output/MatMuloutput/bias/read*'
_output_shapes
:?????????
*
T0*
data_formatNHWC
?
/train/SparseSoftmaxCrossEntropyWithLogits/ShapeShape	network/y*
T0	*
out_type0*#
_output_shapes
:?????????
?
Mtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsnetwork/output/BiasAdd	network/y*6
_output_shapes$
":?????????:?????????
*
Tlabels0	*
T0
U
train/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?

train/lossMeanMtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitstrain/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
w
-train/gradients/train/loss_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
'train/gradients/train/loss_grad/ReshapeReshapetrain/gradients/Fill-train/gradients/train/loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
?
%train/gradients/train/loss_grad/ShapeShapeMtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
?
$train/gradients/train/loss_grad/TileTile'train/gradients/train/loss_grad/Reshape%train/gradients/train/loss_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0
?
'train/gradients/train/loss_grad/Shape_1ShapeMtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
j
'train/gradients/train/loss_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%train/gradients/train/loss_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
$train/gradients/train/loss_grad/ProdProd'train/gradients/train/loss_grad/Shape_1%train/gradients/train/loss_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'train/gradients/train/loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
&train/gradients/train/loss_grad/Prod_1Prod'train/gradients/train/loss_grad/Shape_2'train/gradients/train/loss_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)train/gradients/train/loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
'train/gradients/train/loss_grad/MaximumMaximum&train/gradients/train/loss_grad/Prod_1)train/gradients/train/loss_grad/Maximum/y*
T0*
_output_shapes
: 
?
(train/gradients/train/loss_grad/floordivFloorDiv$train/gradients/train/loss_grad/Prod'train/gradients/train/loss_grad/Maximum*
T0*
_output_shapes
: 
?
$train/gradients/train/loss_grad/CastCast(train/gradients/train/loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
?
'train/gradients/train/loss_grad/truedivRealDiv$train/gradients/train/loss_grad/Tile$train/gradients/train/loss_grad/Cast*
T0*#
_output_shapes
:?????????
?
train/gradients/zeros_like	ZerosLikeOtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:?????????

?
rtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientOtrain/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:?????????
*?
message??Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
?
qtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
?????????*
dtype0
?
mtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims'train/gradients/train/loss_grad/truedivqtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:?????????
?
ftrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulmtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsrtrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:?????????

?
7train/gradients/network/output/BiasAdd_grad/BiasAddGradBiasAddGradftrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
_output_shapes
:
*
T0*
data_formatNHWC
?
<train/gradients/network/output/BiasAdd_grad/tuple/group_depsNoOp8^train/gradients/network/output/BiasAdd_grad/BiasAddGradg^train/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul
?
Dtrain/gradients/network/output/BiasAdd_grad/tuple/control_dependencyIdentityftrain/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul=^train/gradients/network/output/BiasAdd_grad/tuple/group_deps*
T0*y
_classo
mkloc:@train/gradients/train/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*'
_output_shapes
:?????????

?
Ftrain/gradients/network/output/BiasAdd_grad/tuple/control_dependency_1Identity7train/gradients/network/output/BiasAdd_grad/BiasAddGrad=^train/gradients/network/output/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*
T0*J
_class@
><loc:@train/gradients/network/output/BiasAdd_grad/BiasAddGrad
?
1train/gradients/network/output/MatMul_grad/MatMulMatMulDtrain/gradients/network/output/BiasAdd_grad/tuple/control_dependencyoutput/kernel/read*
T0*'
_output_shapes
:?????????d*
transpose_a( *
transpose_b(
?
3train/gradients/network/output/MatMul_grad/MatMul_1MatMulnetwork/h2/ReluDtrain/gradients/network/output/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:d
*
transpose_a(*
transpose_b( 
?
;train/gradients/network/output/MatMul_grad/tuple/group_depsNoOp2^train/gradients/network/output/MatMul_grad/MatMul4^train/gradients/network/output/MatMul_grad/MatMul_1
?
Ctrain/gradients/network/output/MatMul_grad/tuple/control_dependencyIdentity1train/gradients/network/output/MatMul_grad/MatMul<^train/gradients/network/output/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@train/gradients/network/output/MatMul_grad/MatMul*'
_output_shapes
:?????????d
?
Etrain/gradients/network/output/MatMul_grad/tuple/control_dependency_1Identity3train/gradients/network/output/MatMul_grad/MatMul_1<^train/gradients/network/output/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/network/output/MatMul_grad/MatMul_1*
_output_shapes

:d

?
-train/gradients/network/h2/Relu_grad/ReluGradReluGradCtrain/gradients/network/output/MatMul_grad/tuple/control_dependencynetwork/h2/Relu*
T0*'
_output_shapes
:?????????d
?
3train/gradients/network/h2/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/network/h2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
?
8train/gradients/network/h2/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/network/h2/BiasAdd_grad/BiasAddGrad.^train/gradients/network/h2/Relu_grad/ReluGrad
?
@train/gradients/network/h2/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/network/h2/Relu_grad/ReluGrad9^train/gradients/network/h2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????d*
T0*@
_class6
42loc:@train/gradients/network/h2/Relu_grad/ReluGrad
?
Btrain/gradients/network/h2/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/network/h2/BiasAdd_grad/BiasAddGrad9^train/gradients/network/h2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/network/h2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
?
-train/gradients/network/h2/MatMul_grad/MatMulMatMul@train/gradients/network/h2/BiasAdd_grad/tuple/control_dependencyh2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(
?
/train/gradients/network/h2/MatMul_grad/MatMul_1MatMulnetwork/h1/Relu@train/gradients/network/h2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	?d*
transpose_a(
?
7train/gradients/network/h2/MatMul_grad/tuple/group_depsNoOp.^train/gradients/network/h2/MatMul_grad/MatMul0^train/gradients/network/h2/MatMul_grad/MatMul_1
?
?train/gradients/network/h2/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/network/h2/MatMul_grad/MatMul8^train/gradients/network/h2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/network/h2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Atrain/gradients/network/h2/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/network/h2/MatMul_grad/MatMul_18^train/gradients/network/h2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/network/h2/MatMul_grad/MatMul_1*
_output_shapes
:	?d
?
-train/gradients/network/h1/Relu_grad/ReluGradReluGrad?train/gradients/network/h2/MatMul_grad/tuple/control_dependencynetwork/h1/Relu*(
_output_shapes
:??????????*
T0
?
3train/gradients/network/h1/BiasAdd_grad/BiasAddGradBiasAddGrad-train/gradients/network/h1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
8train/gradients/network/h1/BiasAdd_grad/tuple/group_depsNoOp4^train/gradients/network/h1/BiasAdd_grad/BiasAddGrad.^train/gradients/network/h1/Relu_grad/ReluGrad
?
@train/gradients/network/h1/BiasAdd_grad/tuple/control_dependencyIdentity-train/gradients/network/h1/Relu_grad/ReluGrad9^train/gradients/network/h1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/network/h1/Relu_grad/ReluGrad*(
_output_shapes
:??????????
?
Btrain/gradients/network/h1/BiasAdd_grad/tuple/control_dependency_1Identity3train/gradients/network/h1/BiasAdd_grad/BiasAddGrad9^train/gradients/network/h1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*
T0*F
_class<
:8loc:@train/gradients/network/h1/BiasAdd_grad/BiasAddGrad
?
-train/gradients/network/h1/MatMul_grad/MatMulMatMul@train/gradients/network/h1/BiasAdd_grad/tuple/control_dependencyh1/kernel/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(*
T0
?
/train/gradients/network/h1/MatMul_grad/MatMul_1MatMul	network/X@train/gradients/network/h1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
??*
transpose_a(*
transpose_b( *
T0
?
7train/gradients/network/h1/MatMul_grad/tuple/group_depsNoOp.^train/gradients/network/h1/MatMul_grad/MatMul0^train/gradients/network/h1/MatMul_grad/MatMul_1
?
?train/gradients/network/h1/MatMul_grad/tuple/control_dependencyIdentity-train/gradients/network/h1/MatMul_grad/MatMul8^train/gradients/network/h1/MatMul_grad/tuple/group_deps*@
_class6
42loc:@train/gradients/network/h1/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
Atrain/gradients/network/h1/MatMul_grad/tuple/control_dependency_1Identity/train/gradients/network/h1/MatMul_grad/MatMul_18^train/gradients/network/h1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/network/h1/MatMul_grad/MatMul_1* 
_output_shapes
:
??
h
#train/GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
?#<
?
;train/GradientDescent/update_h1/kernel/ApplyGradientDescentApplyGradientDescent	h1/kernel#train/GradientDescent/learning_rateAtrain/gradients/network/h1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@h1/kernel* 
_output_shapes
:
??
?
9train/GradientDescent/update_h1/bias/ApplyGradientDescentApplyGradientDescenth1/bias#train/GradientDescent/learning_rateBtrain/gradients/network/h1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:?*
use_locking( *
T0*
_class
loc:@h1/bias
?
;train/GradientDescent/update_h2/kernel/ApplyGradientDescentApplyGradientDescent	h2/kernel#train/GradientDescent/learning_rateAtrain/gradients/network/h2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@h2/kernel*
_output_shapes
:	?d
?
9train/GradientDescent/update_h2/bias/ApplyGradientDescentApplyGradientDescenth2/bias#train/GradientDescent/learning_rateBtrain/gradients/network/h2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@h2/bias*
_output_shapes
:d*
use_locking( 
?
?train/GradientDescent/update_output/kernel/ApplyGradientDescentApplyGradientDescentoutput/kernel#train/GradientDescent/learning_rateEtrain/gradients/network/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@output/kernel*
_output_shapes

:d

?
=train/GradientDescent/update_output/bias/ApplyGradientDescentApplyGradientDescentoutput/bias#train/GradientDescent/learning_rateFtrain/gradients/network/output/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@output/bias*
_output_shapes
:

?
train/GradientDescentNoOp:^train/GradientDescent/update_h1/bias/ApplyGradientDescent<^train/GradientDescent/update_h1/kernel/ApplyGradientDescent:^train/GradientDescent/update_h2/bias/ApplyGradientDescent<^train/GradientDescent/update_h2/kernel/ApplyGradientDescent>^train/GradientDescent/update_output/bias/ApplyGradientDescent@^train/GradientDescent/update_output/kernel/ApplyGradientDescent
Z
eval/in_top_k/InTopKV2/kConst*
value	B	 R*
dtype0	*
_output_shapes
: 
?
eval/in_top_k/InTopKV2InTopKV2network/output/BiasAdd	network/yeval/in_top_k/InTopKV2/k*
T0	*#
_output_shapes
:?????????
v
	eval/CastCasteval/in_top_k/InTopKV2*

SrcT0
*
Truncate( *#
_output_shapes
:?????????*

DstT0
T

eval/ConstConst*
valueB: *
dtype0*
_output_shapes
:
f
	eval/MeanMean	eval/Cast
eval/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
initNoOp^h1/bias/Assign^h1/kernel/Assign^h2/bias/Assign^h2/kernel/Assign^output/bias/Assign^output/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
?
save/SaveV2/tensor_namesConst*
_output_shapes
:*W
valueNBLBh1/biasB	h1/kernelBh2/biasB	h2/kernelBoutput/biasBoutput/kernel*
dtype0
o
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
valueBB B B B B B *
dtype0
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesh1/bias	h1/kernelh2/bias	h2/kerneloutput/biasoutput/kernel*
dtypes

2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst*W
valueNBLBh1/biasB	h1/kernelBh2/biasB	h2/kernelBoutput/biasBoutput/kernel*
dtype0*
_output_shapes
:
r
save/RestoreV2/shape_and_slicesConst*
valueBB B B B B B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*,
_output_shapes
::::::*
dtypes

2
?
save/AssignAssignh1/biassave/RestoreV2*
use_locking(*
T0*
_class
loc:@h1/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_1Assign	h1/kernelsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@h1/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_2Assignh2/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@h2/bias*
validate_shape(*
_output_shapes
:d
?
save/Assign_3Assign	h2/kernelsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@h2/kernel*
validate_shape(*
_output_shapes
:	?d
?
save/Assign_4Assignoutput/biassave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@output/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_5Assignoutput/kernelsave/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes

:d

v
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_44e75649565b48a5bc5e55c3c09e9567/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst*W
valueNBLBh1/biasB	h1/kernelBh2/biasB	h2/kernelBoutput/biasBoutput/kernel*
dtype0*
_output_shapes
:
q
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:*
valueBB B B B B B *
dtype0
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesh1/bias	h1/kernelh2/bias	h2/kerneloutput/biasoutput/kernel*
dtypes

2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
_output_shapes
:*
T0*

axis 
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:*W
valueNBLBh1/biasB	h1/kernelBh2/biasB	h2/kernelBoutput/biasBoutput/kernel*
dtype0
t
!save_1/RestoreV2/shape_and_slicesConst*
valueBB B B B B B *
dtype0*
_output_shapes
:
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*,
_output_shapes
::::::*
dtypes

2
?
save_1/AssignAssignh1/biassave_1/RestoreV2*
T0*
_class
loc:@h1/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_1Assign	h1/kernelsave_1/RestoreV2:1*
T0*
_class
loc:@h1/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_2Assignh2/biassave_1/RestoreV2:2*
T0*
_class
loc:@h2/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
?
save_1/Assign_3Assign	h2/kernelsave_1/RestoreV2:3*
validate_shape(*
_output_shapes
:	?d*
use_locking(*
T0*
_class
loc:@h2/kernel
?
save_1/Assign_4Assignoutput/biassave_1/RestoreV2:4*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@output/bias
?
save_1/Assign_5Assignoutput/kernelsave_1/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@output/kernel*
validate_shape(*
_output_shapes

:d

?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
trainable_variables??
[
h1/kernel:0h1/kernel/Assignh1/kernel/read:02&h1/kernel/Initializer/random_uniform:08
J
	h1/bias:0h1/bias/Assignh1/bias/read:02h1/bias/Initializer/zeros:08
[
h2/kernel:0h2/kernel/Assignh2/kernel/read:02&h2/kernel/Initializer/random_uniform:08
J
	h2/bias:0h2/bias/Assignh2/bias/read:02h2/bias/Initializer/zeros:08
k
output/kernel:0output/kernel/Assignoutput/kernel/read:02*output/kernel/Initializer/random_uniform:08
Z
output/bias:0output/bias/Assignoutput/bias/read:02output/bias/Initializer/zeros:08"%
train_op

train/GradientDescent"?
	variables??
[
h1/kernel:0h1/kernel/Assignh1/kernel/read:02&h1/kernel/Initializer/random_uniform:08
J
	h1/bias:0h1/bias/Assignh1/bias/read:02h1/bias/Initializer/zeros:08
[
h2/kernel:0h2/kernel/Assignh2/kernel/read:02&h2/kernel/Initializer/random_uniform:08
J
	h2/bias:0h2/bias/Assignh2/bias/read:02h2/bias/Initializer/zeros:08
k
output/kernel:0output/kernel/Assignoutput/kernel/read:02*output/kernel/Initializer/random_uniform:08
Z
output/bias:0output/bias/Assignoutput/bias/read:02output/bias/Initializer/zeros:08*?
serving_default?
,
input#
network/X:0??????????9
output/
network/output/BiasAdd:0?????????
tensorflow/serving/predict