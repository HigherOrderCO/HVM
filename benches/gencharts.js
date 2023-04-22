const { ChartJSNodeCanvas } = require('chartjs-node-canvas');

var results = require("./results.json");

var label_format = {
  "lambda-multiplication": {
    unit: n => String(n) + "m",
    name: "multiplier",
  },
  "sort-bubble": {
    unit: n => String(n) + "k",
    name: "list length",
  },
  "sort-bitonic": {
    unit: n => "2^" + String(n),
    name: "list length",
  },
  "sort-quick": {
    unit: n => "2^" + String(n),
    name: "list length",
  },
  "sort-radix": {
    unit: n => "2^" + String(n),
    name: "list length",
  },
}

var lang_color = {
  "GHC": "purple",
  "HVM-1": "#425F57",
  "HVM-2": "#749F82",
  "HVM-4": "#A8E890",
  "HVM-8": "#CFFF8D",
};

var max_time_limit = {
  //"checker_nat_exp": 160,
  //"checker_nat_exp_church": 28,
  //"checker_tree_fold": 45,
  //"checker_tree_fold_church": 12,
};

var charts = {};
for (var result of results) {
  var chart = result.task.replace("/","-");
  var lang = result.lang;
  var init = Number(result.size);
  var time = Number(result.time);

  if (!charts[chart]) {
    charts[chart] = {};
  }

  if (!charts[chart][lang]) {
    charts[chart][lang] = {
      label: lang,
      data: [],
      init: init,
      borderColor: lang_color[lang],
      fill: false,
    };
  }

  // FIXME: I'm replacing the first value by 0 since it is skewed.
  // Instead, we should perform a dry-run of the first benchmark.
  if (charts[chart][lang].data.length === 0) {
    charts[chart][lang].data.push(0);
  } else {
    charts[chart][lang].data.push(time);
  }
}

for (let chart in charts) {

  var labels = null;
  var datasets = [];

  var max_time = 0;
  if (!max_time_limit[chart]) {
    for (var lang in charts[chart]) {
      for (var time of charts[chart][lang].data) {
        max_time = Math.floor(Math.max(max_time, time));
      }
    }
  } else {
    max_time = max_time_limit[chart];
  }

  for (var lang in charts[chart]) {
    datasets.push(charts[chart][lang]);
    if (!labels) {
      labels = [];
      for (var i = 0; i < charts[chart][lang].data.length; ++i) {
        labels.push(label_format[chart].unit(charts[chart][lang].init + i));
      }
    }
  }

  const configuration = {
    type: 'line',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: chart,
        },
      },
      interaction: {
        intersect: false,
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: label_format[chart].name,
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Time Elapsed (seconds)'
          },
          min: 0,
          max: max_time,
        }
      }
    },
  };

  const width = 1000; //px
  const height = 400; //px
  const backgroundColour = 'white';
  const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height, backgroundColour });

  (async () => {
      const image = await chartJSNodeCanvas.renderToBuffer(configuration);
      require("fs").writeFileSync("_results_/"+chart+".png", image);
      console.log("_results_/"+chart+".png");
      //const dataUrl = await chartJSNodeCanvas.renderToDataURL(configuration);
      //const stream = chartJSNodeCanvas.renderToStream(configuration);
  })();
};
