(function(){
var $=function(i){return document.getElementById(i)};
function pct(x,d){return (x*100).toFixed(d==null?1:d)+"%"}
function kpi(v,l){return '<div class="kpi"><b>'+v+'</b><span>'+l+'</span></div>'}
function label(s){return s.replace(/_/g," ")}
Promise.all([fetch("data.json").then(function(r){return r.json()}),fetch("model.json").then(function(r){return r.json()})]).then(function(a){
var D=a[0],M=a[1],m=D.metrics;
$("kpis").innerHTML=kpi(m.auc_saved,"ROC-AUC, held-out half")+kpi(m.pr_auc_saved,"PR-AUC (base rate "+D.meta.base_rate+"%)")+kpi(m.auc_baseline_lr,"plain logistic regression AUC")+kpi(D.meta.rows.toLocaleString(),"sample rows, "+D.meta.churners+" churners");
$("mraw").textContent=m.mean_raw_score;$("mbase").textContent=D.meta.base_rate+"%";
// top-k
function topk(){var k=+$("k").value;$("kv").textContent=k;var r=D.top_k[k-1];
$("kres").innerHTML=kpi(r[1],"customers contacted")+kpi(r[2],"churners reached")+kpi(pct(r[3]),"precision (random: "+D.meta.base_rate+"%)")+kpi(pct(r[4]),"recall of all churners")+kpi((r[3]/(D.meta.eval_churners/D.meta.eval_rows)).toFixed(2)+"x","lift over random")}
$("k").oninput=topk;topk();
// importance
$("imp").innerHTML=D.importance.slice(0,10).map(function(p){var mx=D.importance[0][1];return '<div class="row"><span class="n">'+label(p[0])+'</span><span class="t"><i style="width:'+(p[1]/mx*100)+'%"></i></span><span class="v">'+p[1]+'%</span></div>'}).join("");
// segments
$("segs").innerHTML=Object.keys(D.segments).map(function(k){var s=D.segments[k],mx=Math.max.apply(null,s.map(function(r){return r[1]}));
return '<div><h3>'+label(k)+'</h3>'+s.map(function(r){return '<div class="row"><span class="n">'+r[0]+' <small>(n='+r[2]+')</small></span><span class="t"><i style="width:'+(r[1]/mx*100)+'%"></i></span><span class="v">'+r[1]+'%</span></div>'}).join("")+'</div>'}).join("");
// svg helpers
function svg(w,h,body){return '<svg viewBox="0 0 '+w+' '+h+'" width="100%">'+body+'</svg>'}
// ROC
var W=360,H=300,P=34,pts=D.roc.map(function(p){return (P+p[0]*(W-P-10)).toFixed(1)+","+(H-P-p[1]*(H-P-10)).toFixed(1)}).join(" ");
$("roc").innerHTML=svg(W,H,'<line class="ax" x1="'+P+'" y1="'+(H-P)+'" x2="'+(W-10)+'" y2="'+(H-P)+'"/><line class="ax" x1="'+P+'" y1="10" x2="'+P+'" y2="'+(H-P)+'"/><line x1="'+P+'" y1="'+(H-P)+'" x2="'+(W-10)+'" y2="10" stroke="#5a3a28" stroke-dasharray="4"/><polyline fill="none" stroke="#ff7a1a" stroke-width="2.5" points="'+pts+'"/><text x="'+(W/2)+'" y="'+(H-8)+'">false positive rate</text><text x="4" y="'+(H/2)+'">TPR</text><text x="'+(P+10)+'" y="30">AUC '+m.auc_saved+' (dashed = chance)</text>');
// calibration
function calrows(rows,title){var mx=0.6;return '<h3>'+title+'</h3>'+rows.map(function(r,i){return '<div class="row"><span class="n">bin '+(i+1)+' (n='+r[2]+')</span><span class="t two"><i class="p" style="width:'+Math.min(100,r[0]/mx*100)+'%"></i><i class="o" style="width:'+Math.min(100,r[1]/mx*100)+'%"></i></span><span class="v">'+pct(r[0],0)+' / '+pct(r[1],0)+'</span></div>'}).join("")}
$("cal").innerHTML='<div class="grid2c"><div>'+calrows(D.reliability_raw,"Raw model score")+'</div><div>'+calrows(D.reliability_calibrated,"After Platt calibration")+'</div></div><p class="fine"><span class="dot p"></span>mean predicted &nbsp; <span class="dot o"></span>observed churn rate &nbsp; (8 equal-count bins, held-out half). Brier score: raw '+m.brier_raw+', calibrated '+m.brier_calibrated+', always-predict-base-rate '+m.brier_base_rate+'.</p>';
// what-if
var F=[["contract","sel",[["Month-to-Month",0],["One Year",1],["Two Year",2]]],["customer_satisfaction","rng",0.5],["num_complaints","rng",1],["num_service_calls","rng",1],["late_payments","rng",1],["tenure","rng",1],["monthlycharges","rng",1],["num_services","rng",1],["has_tech_support","sel",[["No",0],["Yes",1]]],["has_online_security","sel",[["No",0],["Yes",1]]]];
var state={};
$("form").innerHTML=F.map(function(f){var n=f[0],d=M.defaults[n];state[n]=d;
if(f[1]=="sel")return '<label>'+label(n)+'<select data-n="'+n+'">'+f[2].map(function(o){return '<option value="'+o[1]+'"'+(o[1]==d?" selected":"")+'>'+o[0]+'</option>'}).join("")+'</select></label>';
var r=M.ranges[n];return '<label>'+label(n)+'<b id="b_'+n+'">'+d+'</b><input type="range" data-n="'+n+'" min="'+r[0]+'" max="'+r[1]+'" step="'+((r[1]-r[0])/100).toFixed(3)+'" value="'+d+'"></label>'}).join("");
var base=ChurnModel.calibrate(M,ChurnModel.rawScore(M,ChurnModel.vector(M,{})));
$("gbase").textContent="average-customer baseline: "+pct(base);
function upd(){var p=ChurnModel.calibrate(M,ChurnModel.rawScore(M,ChurnModel.vector(M,state)));$("gv").textContent=pct(p);$("gb").style.width=Math.min(100,p*100*2.5)+"%"}
$("form").addEventListener("input",function(e){var n=e.target.dataset.n;state[n]=parseFloat(e.target.value);var b=$("b_"+n);if(b)b.textContent=(+e.target.value).toFixed(1);upd()});upd();
});})();
