* ====================================================================
* Author :	Woong Choi,
* 		Korea Univ, VLSI Signal Processing Lab.
* ====================================================================
.title	'Korea Univ., VLSI SP Lab.'

* --------------------------------------------------------------------
* [ Options ]
* --------------------------------------------------------------------
.option	post probe
.option	modmonte = 1
.option	measdgt = 6
.option	numdgt = 6
.option	mtthred = 2
.option	measfile = 1
.option	measform = 3
.option	mcbrief = 2
.option	abstol=1e-6 reltol=1e-6

* --------------------------------------------------------------------
* [ PVT MODEL ]
* --------------------------------------------------------------------
.hdl	"/home/woong/project/PTM-MG/BSIMCMG107.0.0_20130712/code/bsimcmg.va"
.lib	'/home/woong/project/PTM-MG/mpfp_sram_models' ptm14lstp_mpfp_sram
.inc	'/home/woong/project/sim/python/net/sram_6t_cell/netlist'
.param	p_vdd	= 0.6
.param	p_vss	= 0
.temp		090

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvtpul	= '-scrit_pul*vtsig_pu'
.param	dvtpur	= '+scrit_pur*vtsig_pu'
.param	dvtpgl	= '-scrit_pgl*vtsig_pg'
.param	dvtpgr	= '+scrit_pgr*vtsig_pg'
.param	dvtpdl	= '+scrit_pdl*vtsig_pd'
.param	dvtpdr	= '-scrit_pdr*vtsig_pd'

.param	vtsig_pu = 0.035
.param	vtsig_pg = 0.035
.param	vtsig_pd = 0.035

.param	scrit =       3.942466 
* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	ast_vbs = 1.00
.param	ast_wlu = 0.70
.param	ast_sbl = 1.00

vvdd	vdd	0	'p_vdd*ast_vbs'
vvss	vss	0	p_vss

.param	p_vcrit	= 0
vcrit	q	0	p_vcrit

.nodeset	v(q)	p_vss
.nodeset	v(qb)	p_vdd

vwl		wl		0	'p_vdd*ast_wlu'
vbl		bl		0	'p_vdd*ast_sbl'
vblb	blb		0	'p_vdd*ast_sbl'

.data	sweepvt
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0 0 0
.enddata

.data	sweepvt_pul 
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
   3.942466 0 0 0 0 0
.enddata

.data	sweepvt_pur
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0    3.942466 0 0 0 0
.enddata

.data	sweepvt_pgl
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0    3.942466 0 0 0
.enddata

.data	sweepvt_pgr
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0    3.942466 0 0
.enddata

.data	sweepvt_pdl
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0    3.942466 0
.enddata

.data	sweepvt_pdr
scrit_pul scrit_pur scrit_pgl scrit_pgr scrit_pdl scrit_pdr
0 0 0 0 0    3.942466
.enddata

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt
.meas	dc	icrit_normal	max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pul
.meas	dc	icrit_pul0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pul0		param	('(icrit_normal-icrit_pul0)/scrit')

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pur
.meas	dc	icrit_pur0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pur0		param	('(icrit_normal-icrit_pur0)/scrit')

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pgl
.meas	dc	icrit_pgl0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pgl0		param	('(icrit_normal-icrit_pgl0)/scrit')

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pgr
.meas	dc	icrit_pgr0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pgr0		param	('(icrit_normal-icrit_pgr0)/scrit')

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pdl
.meas	dc	icrit_pdl0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pdl0		param	('(icrit_normal-icrit_pdl0)/scrit')

.dc	p_vcrit	0	vdd	0.001	sweep data = sweepvt_pdr
.meas	dc	icrit_pdr0		max	par('-i(vcrit)')	from = 0 to	= 'p_vdd/2'
.meas	dc	slope_pdr0		param	('(icrit_normal-icrit_pdr0)/scrit')

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
