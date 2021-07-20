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
.option	measform = 3
.option	mcbrief = 2
.option	abstol=1e-6 reltol=1e-6

* --------------------------------------------------------------------
* [ PVT MODEL ]
* --------------------------------------------------------------------
.hdl	"/home/woong/project/PTM-MG/BSIMCMG107.0.0_20130712/code/bsimcmg.va"
.lib	'/home/woong/project/PTM-MG/mpfp_sram_models' ptm14lstp_mpfp_sram
.inc	'/home/woong/project/sim/python/net/sram_6t_cell/netlist'
.param	p_vdd	= {VDD:0.30}
.param	p_vss	= 0
.temp		{TEMP:090}

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvtpul	= '-hvec_pul*vtsig_pu*dcadm'
.param	dvtpur	= '+hvec_pur*vtsig_pu*dcadm'
.param	dvtpgl	= '-hvec_pgl*vtsig_pg*dcadm'
.param	dvtpgr	= '+hvec_pgr*vtsig_pg*dcadm'
.param	dvtpdl	= '+hvec_pdl*vtsig_pd*dcadm'
.param	dvtpdr	= '-hvec_pdr*vtsig_pd*dcadm'

.param	vtsig_pu = {VTSIG_PU:0.035}
.param	vtsig_pg = {VTSIG_PG:0.035}
.param	vtsig_pd = {VTSIG_PD:0.035}

.param	hvec_pul = {HVEC_PUL:@}
.param	hvec_pur = {HVEC_PUR:@}
.param	hvec_pgl = {HVEC_PGL:@}
.param	hvec_pgr = {HVEC_PGR:@}
.param	hvec_pdl = {HVEC_PDL:@}
.param	hvec_pdr = {HVEC_PDR:@}

* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	ast_vbs = {VBS:1.00}
.param	ast_wlu = {WLU:1.00}
.param	ast_sbl = {SBL:1.00}

vvdd	vdd	0	'p_vdd*ast_vbs'
vvss	vss	0	p_vss

.param	p_vcrit	= 0
vcrit	q	0	p_vcrit

.nodeset	v(q)	p_vss
.nodeset	v(qb)	p_vdd

vwl	wl		0	'p_vdd*ast_wlu'
vbl	bl		0	'p_vdd*ast_sbl'
vblb	blb	0	'p_vdd*ast_sbl'

* --------------------------------------------------------------------
* [ Step1	]
* --------------------------------------------------------------------
.model	optmod opt method = bisection Relin = 1e-4 itropt=100
.param	dcadm = opt1 (0, 0, 20)
.dc		p_vcrit	0	p_vdd	0.001	sweep optimize = opt1 result = icrit model = optmod
.probe	v(*)
.probe	i(*)
.meas	dc	icrit	max par('-i(vcrit)') from = 0 to = 'p_vdd/2' goal = 0

.end
