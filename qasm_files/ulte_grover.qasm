OPENQASM 2.0;
include "qelib1.inc";
qreg in2[1];
qreg in3[1];
qreg q2[1];
qreg an[7];
qreg q6[1];
creg c0[2];
h in2[0];
h in3[0];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
ccx an[2],in2[0],an[6];
cx in2[0],an[2];
cx an[0],an[6];
ccx an[2],in3[0],an[6];
cx in3[0],an[2];
cx an[1],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
x an[4];
cx an[4],an[3];
cx in2[0],an[3];
cx in3[0],an[3];
cx an[3],an[5];
x an[6];
x an[5];
ccx an[6],an[5],q2[0];
x an[6];
x an[5];
x q6[0];
h q6[0];
x q2[0];
cx q2[0],q6[0];
x q2[0];
x q2[0];
x q2[0];
x an[5];
x an[6];
x an[5];
x an[6];
cx an[3],an[5];
cx in3[0],an[3];
cx in2[0],an[3];
cx an[4],an[3];
x an[4];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
cx an[1],an[6];
cx in3[0],an[2];
ccx an[2],in3[0],an[6];
cx an[0],an[6];
cx in2[0],an[2];
ccx an[2],in2[0],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
h in2[0];
x in2[0];
h in3[0];
x in3[0];
h in2[0];
cx in3[0],in2[0];
h in2[0];
x in2[0];
x in3[0];
h in2[0];
h in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
ccx an[2],in2[0],an[6];
cx in2[0],an[2];
cx an[0],an[6];
ccx an[2],in3[0],an[6];
cx in3[0],an[2];
cx an[1],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
x an[4];
cx an[4],an[3];
cx in2[0],an[3];
cx in3[0],an[3];
cx an[3],an[5];
x an[6];
x an[5];
x an[6];
x an[5];
x q2[0];
x q2[0];
x q2[0];
x q2[0];
x an[5];
x an[6];
x an[5];
x an[6];
cx an[3],an[5];
cx in3[0],an[3];
cx in2[0],an[3];
cx an[4],an[3];
x an[4];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
cx an[1],an[6];
cx in3[0],an[2];
ccx an[2],in3[0],an[6];
cx an[0],an[6];
cx in2[0],an[2];
ccx an[2],in2[0],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
h in2[0];
x in2[0];
h in3[0];
x in3[0];
h in2[0];
cx in3[0],in2[0];
h in2[0];
x in2[0];
x in3[0];
h in2[0];
h in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
ccx an[2],in2[0],an[6];
cx in2[0],an[2];
cx an[0],an[6];
ccx an[2],in3[0],an[6];
cx in3[0],an[2];
cx an[1],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
x an[4];
cx an[4],an[3];
cx in2[0],an[3];
cx in3[0],an[3];
cx an[3],an[5];
x an[6];
x an[5];
x an[6];
x an[5];
x q2[0];
x q2[0];
x q2[0];
x q2[0];
x an[5];
x an[6];
x an[5];
x an[6];
cx an[3],an[5];
cx in3[0],an[3];
cx in2[0],an[3];
cx an[4],an[3];
x an[4];
x in3[0];
x an[1];
cx in3[0],an[1];
x in3[0];
cx an[1],an[6];
cx in3[0],an[2];
ccx an[2],in3[0],an[6];
cx an[0],an[6];
cx in2[0],an[2];
ccx an[2],in2[0],an[6];
x in3[0];
cx in3[0],an[1];
x an[1];
x in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
h in2[0];
x in2[0];
h in3[0];
x in3[0];
h in2[0];
cx in3[0],in2[0];
h in2[0];
x in2[0];
x in3[0];
h in2[0];
h in3[0];
barrier in2[0],in3[0],q2[0],an[0],an[1],an[2],an[3],an[4],an[5],an[6],q6[0];
measure in2[0] -> c0[0];
measure in3[0] -> c0[1];
