"""
BEEF-vdW parameters
"""
import numpy as np

t = np.array([4.0, 0.0])

x = np.array([
 1.516501714304992365356,
 0.441353209874497942611,
-0.091821352411060291887,
-0.023527543314744041314,
 0.034188284548603550816,
 0.002411870075717384172,
-0.014163813515916020766,
 0.000697589558149178113,
 0.009859205136982565273,
-0.006737855050935187551,
-0.001573330824338589097,
 0.005036146253345903309,
-0.002569472452841069059,
-0.000987495397608761146,
 0.002033722894696920677,
-0.000801871884834044583,
-0.000668807872347525591,
 0.001030936331268264214,
-0.000367383865990214423,
-0.000421363539352619543,
 0.000576160799160517858,
-0.000083465037349510408,
-0.000445844758523195788,
 0.000460129009232047457,
-0.000005231775398304339,
-0.000423957047149510404,
 0.000375019067938866537,
 0.000021149381251344578,
-0.000190491156503997170,
 0.000073843624209823442])

o = range(len(x))
 
c = np.array([
 0.600166476948828631066,
 0.399833523051171368934,
 1.0])

"""
BEEF-vdW ensemble
"""
uiOmega = np.array([
[ 9.238289896663336e-02 , 1.573812432079919e-01 , 1.029935738540308e-01 , 1.366003143143216e-02 , -2.170819634832974e-02 , -1.971473025898487e-03 , 6.694499988752175e-03 , -1.436837103528228e-03 , -1.894288263659829e-03 , 1.620730202731354e-03 , 3.342742083591797e-05 , -8.935288190655010e-04 , 5.660396510944252e-04 , 1.092640880494036e-04 , -3.909536572033999e-04 , 2.271387694573118e-04 , 4.720081507064245e-05 , -1.728805247746040e-04 , 1.161095890105822e-04 , 1.632569772443308e-05 , -9.505329207480296e-05 , 5.966357079138161e-05 , 3.909940118293563e-05 , -9.094078397503243e-05 , 3.979403197298154e-05 , 5.883724662690913e-05 , -8.868728142026543e-05 , 1.649195968392651e-05 , 3.986378541237102e-05 , -2.080734204109696e-05 , -5.210020320050114e-02 ],
[ 1.573812432080020e-01 , 3.194503568212250e-01 , 2.330350019456029e-01 , 3.539526885754365e-02 , -4.398162505429017e-02 , -7.870052015456349e-03 , 1.288386845762548e-02 , -1.452985165647521e-03 , -3.414852982913958e-03 , 2.242106483095301e-03 , 2.411666744826487e-04 , -1.065238741066354e-03 , 4.135880276069384e-04 , 2.536775346693924e-04 , -2.530397572915468e-04 , -5.690638693892032e-05 , 1.673844673999724e-04 , -9.944997873568069e-06 , -1.718953440120930e-04 , 1.760399953825598e-04 , -4.156338135631344e-06 , -1.832004402941794e-04 , 2.147464735651294e-04 , -6.193272093284920e-05 , -1.319710553323893e-04 , 1.948452573660156e-04 , -5.101630490846988e-05 , -9.176394513865211e-05 , 4.717722996545362e-05 , 7.111249931485782e-06 , -1.890906559696380e-02 ],
[ 1.029935738540465e-01 , 2.330350019456185e-01 , 1.906771663140688e-01 , 4.596131842244390e-02 , -2.792908137436464e-02 , -1.240232492150593e-02 , 5.672917933168648e-03 , 1.434385697982085e-03 , -9.455904542077782e-04 , 3.036359098459168e-05 , 1.161828188486106e-04 , 7.937359374341367e-05 , -1.452498186750268e-04 , 1.384058476815110e-05 , 1.530299855805981e-04 , -1.908370243275392e-04 , 5.614920168522352e-05 , 1.448595900033545e-04 , -2.366731351667913e-04 , 1.303628937641542e-04 , 8.403491035544659e-05 , -2.162539474930004e-04 , 1.579894933576654e-04 , 1.853443013110853e-05 , -1.453365923716440e-04 , 1.270119640983266e-04 , 1.393651877686879e-05 , -8.735349638010247e-05 , 1.562163815156337e-05 , 1.819382613180743e-05 , 1.382668594717776e-02 ],
[ 1.366003143144247e-02 , 3.539526885755911e-02 , 4.596131842245237e-02 , 3.412600355844948e-02 , 5.788002236623282e-03 , -9.314441356035262e-03 , -5.276305980529734e-03 , 2.351769282262449e-03 , 1.746899840570664e-03 , -1.053810170761046e-03 , -2.902616086744972e-04 , 5.752547360555607e-04 , -8.857003353891879e-05 , -2.395794347875841e-04 , 1.413569388536142e-04 , 5.605747482892052e-05 , -9.488998643296934e-05 , 2.026963310534137e-05 , 3.772638762355388e-05 , -4.067190865485931e-05 , 1.321492117521963e-05 , 1.940880629107831e-05 , -3.480998018498056e-05 , 1.778991053651829e-05 , 1.586887875776044e-05 , -3.017037178432038e-05 , 6.647594986708508e-06 , 1.545376441325688e-05 , -5.578313586587479e-06 , -2.498675358121092e-06 , -7.076421937394695e-03 ],
[ -2.170819634832771e-02 , -4.398162505428508e-02 , -2.792908137435959e-02 , 5.788002236625639e-03 , 1.599472206930952e-02 , 1.608917143245890e-03 , -5.597384471167169e-03 , -1.499164748509191e-03 , 1.031475806000458e-03 , 5.332996506181574e-04 , -2.489713532023827e-04 , -1.029965243518429e-04 , 1.699409468310518e-04 , -5.189717276078564e-05 , -6.126197146900113e-05 , 8.454620554637730e-05 , -2.898403340456230e-05 , -4.695866195676658e-05 , 7.705234549813160e-05 , -3.658438803802928e-05 , -3.319317982415972e-05 , 6.573717163798472e-05 , -3.698152620572900e-05 , -1.629294629181860e-05 , 4.241341573520274e-05 , -2.624727597577873e-05 , -1.229090821564833e-05 , 2.348090332681114e-05 , -2.215657597169080e-07 , -6.444872622959645e-06 , 7.322667111791249e-04 ],
[ -1.971473025900972e-03 , -7.870052015460869e-03 , -1.240232492150907e-02 , -9.314441356035836e-03 , 1.608917143246348e-03 , 7.634754660592785e-03 , 2.015667017611551e-03 , -3.623574339977459e-03 , -1.474755821692741e-03 , 1.127995802260326e-03 , 4.639737083120432e-04 , -4.567637545650261e-04 , -2.016876766012911e-05 , 2.508509815496272e-04 , -1.147671414054848e-04 , -7.415040892571524e-05 , 9.932046149486572e-05 , -1.325820303664777e-05 , -5.028147494244732e-05 , 4.435536803388949e-05 , -2.227553213442618e-06 , -3.139708798837062e-05 , 3.307650446358692e-05 , -6.558671845195734e-06 , -2.123041867524418e-05 , 2.397646436678162e-05 , 9.138618011606733e-07 , -1.527849014454442e-05 , 2.261408120954423e-06 , 3.617283769859004e-06 , 2.325697711871941e-03 ],
[ 6.694499988750638e-03 , 1.288386845762195e-02 , 5.672917933165492e-03 , -5.276305980530938e-03 , -5.597384471167074e-03 , 2.015667017611739e-03 , 4.377508336814056e-03 , 4.100359917331289e-04 , -1.876150671093797e-03 , -7.271917289430953e-04 , 4.632933527994722e-04 , 2.963398987389869e-04 , -1.506945170950558e-04 , -5.149346314745077e-05 , 9.215110292974351e-05 , -3.132804577761338e-05 , -2.100641270393858e-05 , 3.506730172274297e-05 , -2.465494126635098e-05 , 1.240900749825681e-06 , 2.076535734347166e-05 , -2.285062874633954e-05 , 4.208354769194986e-06 , 1.425348474305690e-05 , -1.526811061895161e-05 , 3.047660598079506e-06 , 9.299255727538788e-06 , -8.183025849838069e-06 , -2.016271133614633e-06 , 3.118202698102115e-06 , -1.983005807705875e-03 ],
[ -1.436837103527614e-03 , -1.452985165646303e-03 , 1.434385697983009e-03 , 2.351769282262657e-03 , -1.499164748509336e-03 , -3.623574339977513e-03 , 4.100359917331572e-04 , 3.388139698932502e-03 , 4.194131188659545e-04 , -1.640686728848097e-03 , -4.535159587025243e-04 , 5.155942974268080e-04 , 1.219637950738874e-04 , -1.881362361335498e-04 , 5.406677887798438e-05 , 6.730117550948196e-05 , -6.826604522477651e-05 , -7.600076704978491e-08 , 4.545041141091276e-05 , -3.434406804211548e-05 , -5.396753498031206e-06 , 3.160900890445868e-05 , -2.489184945477622e-05 , -2.480536094745677e-06 , 2.230938441981598e-05 , -1.767486060639981e-05 , -6.845063675872953e-06 , 1.581526117380142e-05 , 2.198506926484949e-07 , -4.837425950871762e-06 , -2.819410239268639e-05 ],
[ -1.894288263659430e-03 , -3.414852982912986e-03 , -9.455904542068480e-04 , 1.746899840571073e-03 , 1.031475806000471e-03 , -1.474755821692797e-03 , -1.876150671093806e-03 , 4.194131188659666e-04 , 2.016821929004358e-03 , 2.913183096117767e-04 , -1.031831612901280e-03 , -3.523961692265613e-04 , 3.020345263188065e-04 , 1.358462914820522e-04 , -1.115872186939481e-04 , 4.093795217439325e-06 , 4.590005891560275e-05 , -2.788695451888706e-05 , -4.445454868386084e-06 , 1.774618276396958e-05 , -1.122137909788981e-05 , -3.231227423595720e-06 , 1.210473810098234e-05 , -7.926468935313864e-06 , -3.432017428898823e-06 , 8.827938351713780e-06 , -2.192391060027345e-06 , -4.171466247118773e-06 , 1.331053824099077e-06 , 8.121122753847691e-07 , 1.468573793837378e-03 ],
[ 1.620730202730968e-03 , 2.242106483094428e-03 , 3.036359098381830e-05 , -1.053810170761330e-03 , 5.332996506181955e-04 , 1.127995802260379e-03 , -7.271917289430953e-04 , -1.640686728848104e-03 , 2.913183096117794e-04 , 1.618640260028683e-03 , 1.578833514403573e-04 , -8.684832913376226e-04 , -1.835212360942493e-04 , 2.681276727854413e-04 , 3.285354767345348e-05 , -7.506050741939204e-05 , 4.030911032027864e-05 , 1.270499721233960e-05 , -3.550009040339185e-05 , 2.093845130027192e-05 , 6.936412133339431e-06 , -2.092061019101916e-05 , 1.263627438389547e-05 , 5.132905197400893e-06 , -1.410173385828192e-05 , 8.068421998377687e-06 , 6.590533164499491e-06 , -9.628875957888051e-06 , -1.186884523575427e-06 , 3.379003341108947e-06 , -1.318935000558665e-03 ],
[ 3.342742083582248e-05 , 2.411666744824321e-04 , 1.161828188484188e-04 , -2.902616086745682e-04 , -2.489713532023758e-04 , 4.639737083120528e-04 , 4.632933527994702e-04 , -4.535159587025258e-04 , -1.031831612901280e-03 , 1.578833514403571e-04 , 1.126887798536041e-03 , 1.596306400901984e-04 , -6.262219982793480e-04 , -1.832949555936158e-04 , 2.062011811517906e-04 , 5.639579837834072e-05 , -7.429445085205222e-05 , 1.947674856272851e-05 , 2.925850101283131e-05 , -3.392404367734551e-05 , 7.606268115327377e-06 , 1.774935646371143e-05 , -2.076809415497982e-05 , 3.678275105655822e-06 , 1.351664987117452e-05 , -1.391917758734145e-05 , -3.264922954751679e-06 , 1.128720431864021e-05 , -1.552278484090616e-07 , -3.464691582178041e-06 , 2.259380952893320e-04 ],
[ -8.935288190652161e-04 , -1.065238741065750e-03 , 7.937359374391768e-05 , 5.752547360557256e-04 , -1.029965243518811e-04 , -4.567637545650542e-04 , 2.963398987389943e-04 , 5.155942974268113e-04 , -3.523961692265653e-04 , -8.684832913376213e-04 , 1.596306400901987e-04 , 9.274502975544414e-04 , 4.771446682359326e-05 , -5.007069662988802e-04 , -7.942270207742560e-05 , 1.322450571128168e-04 , 2.441262913064850e-05 , -2.756468125262591e-05 , 6.943645566973078e-06 , 1.041750480940249e-05 , -1.187862037244014e-05 , 1.702364109770825e-06 , 7.400825614557900e-06 , -6.767501859886680e-06 , -7.456805310854244e-07 , 5.695968329623519e-06 , -2.204234030240727e-06 , -2.458146094280224e-06 , 1.077364537604088e-06 , 4.312391512705764e-07 , 5.884326361165565e-04 ],
[ 5.660396510942980e-04 , 4.135880276066762e-04 , -1.452498186752349e-04 , -8.857003353897563e-05 , 1.699409468310743e-04 , -2.016876766011903e-05 , -1.506945170950608e-04 , 1.219637950738874e-04 , 3.020345263188087e-04 , -1.835212360942504e-04 , -6.262219982793482e-04 , 4.771446682359360e-05 , 7.353511125371758e-04 , 8.054171359132859e-05 , -4.354044149858314e-04 , -6.575758219487838e-05 , 1.322779340443631e-04 , 4.893233447412187e-06 , -2.860359932846397e-05 , 1.985815168274937e-05 , 1.407122212777636e-06 , -1.355631776270834e-05 , 9.804336837952511e-06 , 1.705077595669618e-06 , -8.448838581047592e-06 , 5.271239541237292e-06 , 3.753161433794400e-06 , -5.679341230392703e-06 , -7.297839478992945e-07 , 1.996414791054073e-06 , -5.689656491774725e-04 ],
[ 1.092640880493588e-04 , 2.536775346692864e-04 , 1.384058476804722e-05 , -2.395794347876363e-04 , -5.189717276079290e-05 , 2.508509815496312e-04 , -5.149346314745000e-05 , -1.881362361335514e-04 , 1.358462914820523e-04 , 2.681276727854418e-04 , -1.832949555936157e-04 , -5.007069662988805e-04 , 8.054171359132875e-05 , 5.670985721529502e-04 , 4.105350281394086e-05 , -3.243779076268346e-04 , -5.693079967475888e-05 , 9.476238507687856e-05 , 1.671992883730651e-05 , -2.625490072653236e-05 , 1.094711235171939e-05 , 8.092095182176009e-06 , -1.368592923368957e-05 , 4.725521343618847e-06 , 6.462723202671019e-06 , -8.176454311340966e-06 , -1.037965911726869e-06 , 5.963104944027835e-06 , -2.287646204875769e-07 , -1.804397982061943e-06 , 6.675499678278738e-05 ],
[ -3.909536572033257e-04 , -2.530397572913827e-04 , 1.530299855807417e-04 , 1.413569388536693e-04 , -6.126197146900289e-05 , -1.147671414054899e-04 , 9.215110292974495e-05 , 5.406677887798494e-05 , -1.115872186939490e-04 , 3.285354767345385e-05 , 2.062011811517907e-04 , -7.942270207742549e-05 , -4.354044149858315e-04 , 4.105350281394089e-05 , 5.023053531078210e-04 , 1.395753202566780e-05 , -2.794248341066854e-04 , -2.462616877967573e-05 , 7.014950575686348e-05 , 7.678983396148418e-06 , -1.200073137869544e-05 , 4.735853628377502e-06 , 3.823008200476699e-06 , -5.632608045337210e-06 , 1.401726052082347e-06 , 2.631914429094741e-06 , -1.879900165857796e-06 , -6.802392260490853e-07 , 6.412891565621652e-07 , 5.793723170821993e-08 , 2.979440856739876e-04 ],
[ 2.271387694572524e-04 , -5.690638693903491e-05 , -1.908370243276230e-04 , 5.605747482890452e-05 , 8.454620554639201e-05 , -7.415040892571150e-05 , -3.132804577761707e-05 , 6.730117550948228e-05 , 4.093795217440853e-06 , -7.506050741939299e-05 , 5.639579837834042e-05 , 1.322450571128173e-04 , -6.575758219487839e-05 , -3.243779076268348e-04 , 1.395753202566789e-05 , 4.086277915281374e-04 , 2.438181614175771e-05 , -2.406201469878970e-04 , -2.063418073175250e-05 , 6.468348516289834e-05 , 1.651842998945461e-06 , -1.016330205472771e-05 , 7.380837404491689e-06 , 7.876901704903023e-07 , -5.693055610174383e-06 , 3.898194171094561e-06 , 1.890193310260514e-06 , -3.494268997347222e-06 , -2.097250054628417e-07 , 1.107934512468949e-06 , -2.578053969849174e-04 ],
[ 4.720081507065945e-05 , 1.673844673999971e-04 , 5.614920168523253e-05 , -9.488998643297809e-05 , -2.898403340457248e-05 , 9.932046149486507e-05 , -2.100641270393638e-05 , -6.826604522477717e-05 , 4.590005891560220e-05 , 4.030911032027912e-05 , -7.429445085205212e-05 , 2.441262913064812e-05 , 1.322779340443633e-04 , -5.693079967475883e-05 , -2.794248341066855e-04 , 2.438181614175779e-05 , 3.367003211899217e-04 , 1.421493027932063e-05 , -1.961053122230117e-04 , -1.831760815509797e-05 , 5.249705849097755e-05 , 4.009767661794436e-06 , -9.222615132968448e-06 , 4.447935971545765e-06 , 2.844605015203588e-06 , -4.927439995523699e-06 , 2.779858179450743e-07 , 2.890920446156232e-06 , -3.536840533005166e-07 , -7.989052895188473e-07 , -2.873774500946350e-05 ],
[ -1.728805247745767e-04 , -9.944997873510153e-06 , 1.448595900034050e-04 , 2.026963310536173e-05 , -4.695866195676680e-05 , -1.325820303664937e-05 , 3.506730172274367e-05 , -7.600076704937241e-08 , -2.788695451888763e-05 , 1.270499721233979e-05 , 1.947674856272868e-05 , -2.756468125262590e-05 , 4.893233447412072e-06 , 9.476238507687867e-05 , -2.462616877967574e-05 , -2.406201469878971e-04 , 1.421493027932067e-05 , 2.919803798609199e-04 , 7.292181033176667e-06 , -1.680274842794751e-04 , -1.103641130738799e-05 , 4.275283346882578e-05 , 1.839573029824585e-06 , -5.092906646915116e-06 , 2.996296133918005e-06 , 5.026786485483826e-07 , -1.803524706078249e-06 , 7.612853881615933e-07 , 3.175194859018497e-07 , -2.524196216716103e-07 , 2.671139718648832e-04 ],
[ 1.161095890105204e-04 , -1.718953440122134e-04 , -2.366731351668826e-04 , 3.772638762353110e-05 , 7.705234549814230e-05 , -5.028147494244480e-05 , -2.465494126635465e-05 , 4.545041141091324e-05 , -4.445454868384867e-06 , -3.550009040339265e-05 , 2.925850101283112e-05 , 6.943645566973460e-06 , -2.860359932846412e-05 , 1.671992883730641e-05 , 7.014950575686358e-05 , -2.063418073175254e-05 , -1.961053122230117e-04 , 7.292181033176704e-06 , 2.476672606367232e-04 , 8.122604369362667e-06 , -1.452133704846186e-04 , -9.497391478575562e-06 , 3.809665940899583e-05 , 1.059672833862896e-06 , -5.566702444135148e-06 , 4.241342392780321e-06 , 1.125163314158913e-06 , -3.300826353062116e-06 , 2.381295916739009e-07 , 8.492464195141368e-07 , -2.789569803656198e-04 ],
[ 1.632569772446249e-05 , 1.760399953826087e-04 , 1.303628937641828e-04 , -4.067190865486029e-05 , -3.658438803803874e-05 , 4.435536803388934e-05 , 1.240900749828609e-06 , -3.434406804211623e-05 , 1.774618276396873e-05 , 2.093845130027264e-05 , -3.392404367734537e-05 , 1.041750480940207e-05 , 1.985815168274956e-05 , -2.625490072653231e-05 , 7.678983396148288e-06 , 6.468348516289841e-05 , -1.831760815509795e-05 , -1.680274842794751e-04 , 8.122604369362710e-06 , 2.112966630126243e-04 , 5.363176092207731e-06 , -1.235778898069599e-04 , -7.709953870959738e-06 , 3.098655427549614e-05 , 2.634638058314591e-06 , -4.584365006125596e-06 , 7.784307399132289e-07 , 2.345452381285535e-06 , -6.188482408032955e-07 , -4.998403651495349e-07 , 8.079312086264899e-05 ],
[ -9.505329207477657e-05 , -4.156338135574478e-06 , 8.403491035549607e-05 , 1.321492117523870e-05 , -3.319317982416059e-05 , -2.227553213444590e-06 , 2.076535734347213e-05 , -5.396753498031014e-06 , -1.122137909789006e-05 , 6.936412133339521e-06 , 7.606268115327406e-06 , -1.187862037244012e-05 , 1.407122212777626e-06 , 1.094711235171940e-05 , -1.200073137869545e-05 , 1.651842998945439e-06 , 5.249705849097757e-05 , -1.103641130738799e-05 , -1.452133704846186e-04 , 5.363176092207760e-06 , 1.841513653060571e-04 , 4.008684964031859e-06 , -1.088327175419565e-04 , -4.436272922923257e-06 , 2.663616882515994e-05 , 4.441129647729434e-07 , -1.823900470977472e-06 , 9.131027910925659e-07 , 3.423181895869568e-07 , -3.248030257457939e-07 , 1.565114731653676e-04 ],
[ 5.966357079134110e-05 , -1.832004402942522e-04 , -2.162539474930512e-04 , 1.940880629106866e-05 , 6.573717163799288e-05 , -3.139708798836991e-05 , -2.285062874634257e-05 , 3.160900890445919e-05 , -3.231227423594649e-06 , -2.092061019101990e-05 , 1.774935646371122e-05 , 1.702364109771204e-06 , -1.355631776270847e-05 , 8.092095182175919e-06 , 4.735853628377626e-06 , -1.016330205472776e-05 , 4.009767661794407e-06 , 4.275283346882582e-05 , -9.497391478575592e-06 , -1.235778898069599e-04 , 4.008684964031889e-06 , 1.585945240480566e-04 , 4.814276592252276e-06 , -9.505942249560426e-05 , -5.269885642910686e-06 , 2.508762233822088e-05 , 1.002347324957512e-06 , -3.233685256439425e-06 , 3.615248228908033e-07 , 7.731232588721100e-07 , -2.364008973553363e-04 ],
[ 3.909940118295615e-05 , 2.147464735651595e-04 , 1.579894933576790e-04 , -3.480998018498535e-05 , -3.698152620573602e-05 , 3.307650446358831e-05 , 4.208354769197900e-06 , -2.489184945477703e-05 , 1.210473810098150e-05 , 1.263627438389614e-05 , -2.076809415497966e-05 , 7.400825614557483e-06 , 9.804336837952683e-06 , -1.368592923368950e-05 , 3.823008200476585e-06 , 7.380837404491765e-06 , -9.222615132968445e-06 , 1.839573029824542e-06 , 3.809665940899589e-05 , -7.709953870959746e-06 , -1.088327175419565e-04 , 4.814276592252303e-06 , 1.387884209137800e-04 , 2.113244593212237e-06 , -8.153912579909763e-05 , -4.652337820383065e-06 , 1.937304772679640e-05 , 2.478096542996087e-06 , -8.169606503678209e-07 , -4.287488876009555e-07 , 1.035998031439656e-04 ],
[ -9.094078397502061e-05 , -6.193272093282151e-05 , 1.853443013113500e-05 , 1.778991053653038e-05 , -1.629294629181825e-05 , -6.558671845197636e-06 , 1.425348474305646e-05 , -2.480536094745301e-06 , -7.926468935313898e-06 , 5.132905197400817e-06 , 3.678275105655839e-06 , -6.767501859886567e-06 , 1.705077595669545e-06 , 4.725521343618848e-06 , -5.632608045337194e-06 , 7.876901704902667e-07 , 4.447935971545785e-06 , -5.092906646915108e-06 , 1.059672833862867e-06 , 3.098655427549616e-05 , -4.436272922923254e-06 , -9.505942249560430e-05 , 2.113244593212259e-06 , 1.241068277448159e-04 , 1.324825159079387e-06 , -7.356715084057034e-05 , -1.785631352650215e-06 , 1.695100826863567e-05 , 5.774682432637083e-07 , -3.303613432465353e-07 , 9.651449332646128e-05 ],
[ 3.979403197295345e-05 , -1.319710553324410e-04 , -1.453365923716808e-04 , 1.586887875775279e-05 , 4.241341573520792e-05 , -2.123041867524383e-05 , -1.526811061895372e-05 , 2.230938441981634e-05 , -3.432017428898139e-06 , -1.410173385828241e-05 , 1.351664987117440e-05 , -7.456805310851761e-07 , -8.448838581047687e-06 , 6.462723202670970e-06 , 1.401726052082422e-06 , -5.693055610174417e-06 , 2.844605015203572e-06 , 2.996296133918029e-06 , -5.566702444135167e-06 , 2.634638058314581e-06 , 2.663616882515997e-05 , -5.269885642910686e-06 , -8.153912579909767e-05 , 1.324825159079404e-06 , 1.082133675166925e-04 , 2.990415878922840e-06 , -6.513246311773947e-05 , -2.759724213714544e-06 , 1.484095638923724e-05 , 7.424809301046746e-07 , -1.617594954504215e-04 ],
[ 5.883724662691994e-05 , 1.948452573660281e-04 , 1.270119640983281e-04 , -3.017037178432670e-05 , -2.624727597578309e-05 , 2.397646436678337e-05 , 3.047660598081647e-06 , -1.767486060640050e-05 , 8.827938351713212e-06 , 8.068421998378197e-06 , -1.391917758734134e-05 , 5.695968329623178e-06 , 5.271239541237441e-06 , -8.176454311340913e-06 , 2.631914429094653e-06 , 3.898194171094623e-06 , -4.927439995523706e-06 , 5.026786485483527e-07 , 4.241342392780371e-06 , -4.584365006125614e-06 , 4.441129647729196e-07 , 2.508762233822091e-05 , -4.652337820383076e-06 , -7.356715084057034e-05 , 2.990415878922861e-06 , 9.541694080046339e-05 , 5.311088722428387e-07 , -5.655395254747548e-05 , -7.544356044794082e-07 , 1.269980847624510e-05 , 4.696018935268347e-05 ],
[ -8.868728142024831e-05 , -5.101630490843126e-05 , 1.393651877690296e-05 , 6.647594986721235e-06 , -1.229090821564965e-05 , 9.138618011586676e-07 , 9.299255727538887e-06 , -6.845063675872692e-06 , -2.192391060027468e-06 , 6.590533164499501e-06 , -3.264922954751675e-06 , -2.204234030240666e-06 , 3.753161433794360e-06 , -1.037965911726866e-06 , -1.879900165857787e-06 , 1.890193310260486e-06 , 2.779858179450956e-07 , -1.803524706078243e-06 , 1.125163314158881e-06 , 7.784307399132557e-07 , -1.823900470977467e-06 , 1.002347324957483e-06 , 1.937304772679643e-05 , -1.785631352650217e-06 , -6.513246311773947e-05 , 5.311088722428587e-07 , 7.440208775369848e-05 , 7.311641032314037e-07 , -2.774078047441206e-05 , -4.408828958294675e-07 , 1.075017250578020e-04 ],
[ 1.649195968391140e-05 , -9.176394513867907e-05 , -8.735349638012086e-05 , 1.545376441325374e-05 , 2.348090332681419e-05 , -1.527849014454438e-05 , -8.183025849839297e-06 , 1.581526117380169e-05 , -4.171466247118380e-06 , -9.628875957888362e-06 , 1.128720431864013e-05 , -2.458146094280058e-06 , -5.679341230392763e-06 , 5.963104944027804e-06 , -6.802392260490372e-07 , -3.494268997347246e-06 , 2.890920446156225e-06 , 7.612853881616096e-07 , -3.300826353062134e-06 , 2.345452381285531e-06 , 9.131027910925789e-07 , -3.233685256439427e-06 , 2.478096542996079e-06 , 1.695100826863569e-05 , -2.759724213714545e-06 , -5.655395254747549e-05 , 7.311641032314153e-07 , 6.559666484932615e-05 , 1.240877065411180e-07 , -2.470688255280269e-05 , -9.189338863514660e-05 ],
[ 3.986378541236639e-05 , 4.717722996544147e-05 , 1.562163815155139e-05 , -5.578313586592747e-06 , -2.215657597169136e-07 , 2.261408120955417e-06 , -2.016271133614381e-06 , 2.198506926483088e-07 , 1.331053824099042e-06 , -1.186884523575363e-06 , -1.552278484090472e-07 , 1.077364537604021e-06 , -7.297839478992591e-07 , -2.287646204875707e-07 , 6.412891565621495e-07 , -2.097250054628229e-07 , -3.536840533005254e-07 , 3.175194859018434e-07 , 2.381295916739206e-07 , -6.188482408033085e-07 , 3.423181895869513e-07 , 3.615248228908187e-07 , -8.169606503678325e-07 , 5.774682432637071e-07 , 1.484095638923725e-05 , -7.544356044794156e-07 , -2.774078047441205e-05 , 1.240877065411238e-07 , 1.330905767924987e-05 , 8.884104622005010e-08 , -3.158609279173533e-05 ],
[ -2.080734204109082e-05 , 7.111249931498269e-06 , 1.819382613181743e-05 , -2.498675358118083e-06 , -6.444872622960494e-06 , 3.617283769858598e-06 , 3.118202698102355e-06 , -4.837425950871769e-06 , 8.121122753846729e-07 , 3.379003341109011e-06 , -3.464691582178025e-06 , 4.312391512705559e-07 , 1.996414791054076e-06 , -1.804397982061937e-06 , 5.793723170821257e-08 , 1.107934512468949e-06 , -7.989052895188420e-07 , -2.524196216716127e-07 , 8.492464195141338e-07 , -4.998403651495291e-07 , -3.248030257457955e-07 , 7.731232588721048e-07 , -4.287488876009484e-07 , -3.303613432465375e-07 , 7.424809301046709e-07 , 1.269980847624510e-05 , -4.408828958294696e-07 , -2.470688255280269e-05 , 8.884104622005171e-08 , 1.197542910948322e-05 , 3.878501241188344e-05 ],
[ -5.210020320049051e-02 , -1.890906559693971e-02 , 1.382668594719924e-02 , -7.076421937386331e-03 , 7.322667111787697e-04 , 2.325697711870943e-03 , -1.983005807705755e-03 , -2.819410239254837e-05 , 1.468573793837301e-03 , -1.318935000558654e-03 , 2.259380952893342e-04 , 5.884326361165944e-04 , -5.689656491774901e-04 , 6.675499678278620e-05 , 2.979440856739906e-04 , -2.578053969849344e-04 , -2.873774500945195e-05 , 2.671139718648887e-04 , -2.789569803656384e-04 , 8.079312086266559e-05 , 1.565114731653709e-04 , -2.364008973553556e-04 , 1.035998031439817e-04 , 9.651449332646111e-05 , -1.617594954504337e-04 , 4.696018935269557e-05 , 1.075017250578020e-04 , -9.189338863515410e-05 , -3.158609279173351e-05 , 3.878501241188487e-05 , 2.121632678397157e-01 ]])
