/* Pure scoring code, shared by the browser and the Node parity test. */
(function (root) {
  function treeScore(t, x) {
    var i = 0;
    while (t.f[i] >= 0) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
    return t.v[i];
  }
  function rawScore(model, x) {
    var s = 0;
    for (var k = 0; k < model.trees.length; k++) s += treeScore(model.trees[k], x);
    return 1 / (1 + Math.exp(-s));
  }
  function calibrate(model, raw) {
    var p = Math.min(Math.max(raw, 1e-6), 1 - 1e-6);
    var z = model.calibrator.a * Math.log(p / (1 - p)) + model.calibrator.b;
    return 1 / (1 + Math.exp(-z));
  }
  /* real-unit input object -> model feature vector (standardises the 14 scaled columns) */
  function vector(model, real) {
    return model.features.map(function (name) {
      var v = real[name] !== undefined ? real[name] : model.defaults[name];
      var zi = model.zcols.indexOf(name);
      return zi >= 0 ? (v - model.mean[zi]) / model.scale[zi] : v;
    });
  }
  var api = { rawScore: rawScore, calibrate: calibrate, vector: vector };
  if (typeof module !== "undefined") module.exports = api; else root.ChurnModel = api;
})(this);
