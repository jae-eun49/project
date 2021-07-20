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
.param	dvt_pul	= '-scrit_pul*vtsig_pu'
.param	dvt_pur	= '+scrit_pur*vtsig_pu'
.param	dvt_pgl	= '-scrit_pgl*vtsig_pg'
.param	dvt_pgr	= '+scrit_pgr*vtsig_pg'
.param	dvt_pdl	= '+scrit_pdl*vtsig_pd'
.param	dvt_pdr	= '-scrit_pdr*vtsig_pd'

.param	vtsig_pu = 0.035
.param	vtsig_pg = 0.035
.param	vtsig_pd = 0.035

.param	scrit =       5.311808 
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

.data	sweepvt
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0 0 0
.enddata

.data	sweepvt_pul 
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
   5.311808 0 0 0 0 0
.enddata

.data	sweepvt_pur
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0    5.311808 0 0 0 0
.enddata

.data	sweepvt_pgl
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0    5.311808 0 0 0
.enddata

.data	sweepvt_pgr
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0    5.311808 0 0
.enddata

.data	sweepvt_pdl
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0    5.311808 0
.enddata

.data	sweepvt_pdr
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0 0    5.311808
.enddata

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt
.meas	dc	icrit_normal	max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pul
.meas	dc	icrit_pul		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pul0		param	('(icrit_normal-icrit_pul)/scrit')

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pur
.meas	dc	icrit_pur		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pur0		param	('(icrit_normal-icrit_pur)/scrit')

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pgl
.meas	dc	icrit_pgl		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pgl0		param	('(icrit_normal-icrit_pgl)/scrit')

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pgr
.meas	dc	icrit_pgr		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pgr0		param	('(icrit_normal-icrit_pgr)/scrit')

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pdl
.meas	dc	icrit_pdl		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pdl0		param	('(icrit_normal-icrit_pdl)/scrit')

.dc	p_vcrit	0	p_vdd	0.001	sweep data = sweepvt_pdr
.meas	dc	icrit_pdr		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pdr0		param	('(icrit_normal-icrit_pdr)/scrit')

.meas	dc	icrit_ref		param	icrit_normal
.meas	dc	slope_pul		param	slope_pul0
.meas	dc	slope_pur		param	slope_pur0
.meas	dc	slope_pgl		param	slope_pgl0
.meas	dc	slope_pgr		param	slope_pgr0
.meas	dc	slope_pdl		param	slope_pdl0
.meas	dc	slope_pdr		param	slope_pdr0

*calculate garadients etc
.meas	sumslope	param = 'slope_pul + slope_pur + slope_pgl + slope_pgr + slope_pdl + slope_pdr'
.meas	sumsqr		param = 'pow(slope_pul,2)+pow(slope_pur,2)+pow(slope_pgl,2)+pow(slope_pgr,2)+pow(slope_pdl,2)+pow(slope_pdr,2)'
.meas	sqrtsumsqr	param = '(sqrt(sumsqr))'
.meas	vecnat		param = '(sqrtsumsqr/sumslope)'

*calculate most problable vector
.meas	vecmp_pul	param = '(slope_pul/sqrtsumsqr)'
.meas	vecmp_pur	param = '(slope_pur/sqrtsumsqr)'
.meas	vecmp_pgl	param = '(slope_pgl/sqrtsumsqr)'
.meas	vecmp_pgr	param = '(slope_pgr/sqrtsumsqr)'
.meas	vecmp_pdl	param = '(slope_pdl/sqrtsumsqr)'
.meas	vecmp_pdr	param = '(slope_pdr/sqrtsumsqr)'

*calculate hybrid vector
.param	vecmp_ratio = 0.8

.meas	vechybrid_pul	param = 'vecmp_ratio * vecmp_pul + (1-vecmp_ratio) * vecnat'
.meas	vechybrid_pur	param = 'vecmp_ratio * vecmp_pur + (1-vecmp_ratio) * vecnat'
.meas	vechybrid_pgl	param = 'vecmp_ratio * vecmp_pgl + (1-vecmp_ratio) * vecnat'
.meas	vechybrid_pgr	param = 'vecmp_ratio * vecmp_pgr + (1-vecmp_ratio) * vecnat'
.meas	vechybrid_pdl	param = 'vecmp_ratio * vecmp_pdl + (1-vecmp_ratio) * vecnat'
.meas	vechybrid_pdr	param = 'vecmp_ratio * vecmp_pdr + (1-vecmp_ratio) * vecnat'

.end
