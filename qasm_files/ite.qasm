OPENQASM 2.0;
include "qelib1.inc";
qreg q2[1];
qreg q3[1];
qreg q4[1];
qreg q5[1];
qreg q6[1];
h q2[0];
h q3[0];
h q4[0];
ccx q2[0],q3[0],q5[0];
x q2[0];
ccx q2[0],q4[0],q5[0];
x q2[0];
x q5[0];
cx q5[0],q6[0];
x q5[0];
x q6[0];
