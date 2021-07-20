* ====================================================================
* Author :	Woong Choi, woongchoi@sm.ac.kr
* 		Sookmyung Women's Univ, VLSISYS Lab.
* ====================================================================
.title	'SMU, VLSISYS Lab.'
.option	post
.option	probe
.option	measdgt = 6
.option	numdgt = 6
.option	ingold
.option	modmonte = 1
.option	mtthred = 2
.option	measform = 3
.option	mcbrief = 2
.option	abstol=1e-6 reltol=1e-6
*.option ba_file = ""

* --------------------------------------------------------------------
* [ PVT MODEL ]
* --------------------------------------------------------------------
.lib	'/setup/PDK/S28LPP_2107/FullCustom/LN28LPP_HSPICE_S00-V2.0.1.2/HSPICE/LN28LPP_Hspice.lib' NN

.param	p_vdd	= 0.9
.param	p_vss	= 0
.temp	25

* --------------------------------------------------------------------
* [ Include Files]
* --------------------------------------------------------------------
.inc	'/home/woong/test/netlist'

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvt_pul	= '-sigval*vtsig_pu'
.param	dvt_pur	= '+sigval*vtsig_pu'
.param	dvt_pgl	= '-sigval*vtsig_pg'
.param	dvt_pgr	= '+sigval*vtsig_pg'
.param	dvt_pdl	= '+sigval*vtsig_pd'
.param	dvt_pdr	= '-sigval*vtsig_pd'

.param	vtsig_pu = 0.035
.param	vtsig_pg = 0.035
.param	vtsig_pd = 0.035

* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	ast_vbs = 1.0
.param	ast_wlu = 0.7
.param	ast_sbl = 1.0
.param	p_vcrit	= 0

vvdd	vdd	0	'p_vdd*ast_vbs'
vvss	vss	0	p_vss

vcrit	q	0	p_vcrit

.nodeset	v(q)	p_vss
.nodeset	v(qb)	p_vdd

vwl		wl		0	'p_vdd*ast_wlu'
vblt	blt		0	'p_vdd*ast_sbl'
vblb	blb		0	'p_vdd*ast_sbl'

* --------------------------------------------------------------------
* [ Step1	]
* --------------------------------------------------------------------
.model	optmod opt method = bisection Relin = 1e-4 itropt=100
.param	sigval = opt1 (0, 0, 10)
.dc		p_vcrit	0	p_vdd	0.001	sweep optimize = opt1 result = icrit model = optmod
.meas	icrit		max par('-i(vcrit)') from = 0 to = 'p_vdd/2' goal = 0
.meas	vflip		find		v(qb)	when v(qb) = v(q)
.meas	v0			find		v(q)	when par('-i(vcrit)') = 0 rise = 1
.meas	vtrip		find		v(q)	when v(q) = v(qb)

*.dc		p_vcrit	0	p_vdd	0.001	
*.meas	icrit		max par('-i(vcrit)') from = 0 to = 'p_vdd/2' goal = 0

.probe	v(*)
.probe	i(*)

.end
