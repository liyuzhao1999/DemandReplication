cd "/Users/liyu/Dropbox/Replication/Tuchman2019/Replication"
********************************************************************************
* 1 - Import and Save Data
********************************************************************************
* column 2: homogeneous model with state dependence
import delimited "data/delta_homo.csv", clear
save data/deltahat_homo.dta, replace

* column 3: heterogeneous model without state dependence
import delimited "data/delta_heter_woG.csv", clear
save data/deltahat_S9.dta, replace

* column 4: heterogeneous model with state dependence
import delimited "data/delta_heter_withG.csv", clear
save data/deltahat_S3.dta, replace


********************************************************************************
* 2 - Analyze Data
********************************************************************************

use data/ModelData.dta, clear

keep _border dma_code _dma_border dma_descr week date grp_ecig* grp_anticig* price_ecig_ct_fill ///
	price_pack ecigin share_cigs_norm share_ecig_norm lgrp_* share_cess_norm price_cess_fill
sort _dma_border date

* select which of the 3 models to estimate. uncomment the one you want to run
*merge 1:1 _n using Data/deltahat_G2.dta, nogen // table 6, column 2: homogeneous model with state dependence
*merge 1:1 _n using Data/deltahat_S9.dta, nogen // table 6, column 3: heterogeneous model without state dependence
merge 1:1 dma_code _border week using Data/deltahat_homo.dta, nogen // table 6, column 4: heterogeneous model with state dependence


rename price_pack price1
rename price_cess_ price2
rename price_ecig_ price3
rename delta_cig delta1
rename delta_cess delta2
rename delta_ecig delta3
rename share_cigs_norm share1
rename share_cess_norm share2
rename share_ecig_norm share3

reshape long price delta share, i(_border _dma_border dma_descr week date grp_ecig_tot grp_anticig_tot) j(product)

drop if product == 3 & ecigin == 0


* create ad variables for estimation with cessation products 
gen grp_ecig_tot_c = grp_ecig_tot * (product == 1)
gen grp_ecig_tot_q = grp_ecig_tot * (product == 2)
gen grp_ecig_tot_e = grp_ecig_tot * (product == 3)
gen grp_anticig_tot_q = grp_anticig_tot*(product == 2)

gen lgrp_ecig_tot_c = log(1+grp_ecig_tot_c)
gen lgrp_ecig_tot_q = log(1+grp_ecig_tot_q)
gen lgrp_ecig_tot_e = log(1+grp_ecig_tot_e)
gen lgrp_anticig_tot_q = log(1+grp_anticig_tot_q)


* create product specific fixed effects for border strategy
egen _dma_border_j = group(_dma_border product)

gen month_n = month(date)+(12*(year(date)-2010))
egen _month_border_j = group(_border month_n product)


* ESTIMATE FINAL SPECIFICATION FOR THE PAPER******
reghdfe delta price lgrp_ecig_tot_c lgrp_ecig_tot_q lgrp_ecig_tot_e lgrp_anticig_tot_q , a(_dma_border_j _month_border_j)
