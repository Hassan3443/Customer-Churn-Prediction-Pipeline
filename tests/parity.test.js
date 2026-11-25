const fs=require("fs"),path=require("path"),assert=require("assert");
const C=require("../docs/model.js");
const M=JSON.parse(fs.readFileSync(path.join(__dirname,"../docs/model.json")));
const P=JSON.parse(fs.readFileSync(path.join(__dirname,"fixtures/parity.json")));
let worst=0;
for(const r of P){const raw=C.rawScore(M,r.x),cal=C.calibrate(M,raw);worst=Math.max(worst,Math.abs(raw-r.raw),Math.abs(cal-r.cal));}
assert(worst<1e-9,"parity "+worst);
// vector(): real-unit input round-trips to z-space
const x=P[0].x,real={};M.zcols.forEach((c,i)=>{real[c]=x[M.features.indexOf(c)]*M.scale[i]+M.mean[i]});
const v=C.vector(M,real);v.forEach((a,i)=>{if(M.zcols.includes(M.features[i]))assert(Math.abs(a-x[i])<1e-9)});
console.log("node parity ok on",P.length,"rows, worst diff",worst);
