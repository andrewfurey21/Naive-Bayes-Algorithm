let data = require('./data.json');
let size = Object.keys(data[0]).length - 1;
let k = 5;

//Calculates variance
function calcVariance(arr){
  let mean = 0;
  for (let i = 0; i < arr.length; i++){
  	mean += arr[i];
  }
  mean /= arr.length;
  let arr2 = [];
  for (let i = 0; i < arr.length; i++){
  	arr2.push(Math.pow(arr[i] - mean, 2));
  }

  let sum = 0;
  for (let i = 0; i < arr.length; i++){
  	sum += arr2[i];
  }
  let variance = sum / (arr.length - 1);
  return variance;
}

// Shuffles array
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
      let j = Math.floor(Math.random() * (i + 1));
      let temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
	return array;
}

//Naive Bayes Class
class NaiveBayes {
  constructor(n_features, n_classes) {
    this.n_features = n_features;
    this.n_classes = n_classes;
    this.probs = new Array(n_classes);
    this.probs.fill(0);

    this.featureMeans = [];
    this.featureVariances = [];
    for (let i = 0; i < this.n_classes; i++){

      this.featureMeans.push(new Array(this.n_features));
      this.featureVariances.push(new Array(this.n_features));

      for (let j = 0; j < this.n_features; j++){
        this.featureMeans[i].fill(0);
        this.featureVariances[i].fill(0);
    }
  }
}

  train(samples){

    let d = 0;

    for (let i = 0; i < samples.length; i++){
      d = samples[i].diagnosis;
      this.probs[d]++;

      for (let j = 0; j < size; j++){
        this.featureMeans[d][j] += index_sample(samples[i], j);
      }
    }

    for (let i = 0; i < this.n_classes; i++){
      for (let j = 0; j < this.n_features; j++){
        this.featureMeans[i][j] /= this.probs[i];
      }
    }

    for (let i = 0; i < samples.length; i++){
      d = samples[i].diagnosis;
      for (let j = 0; j < size; j++){
        this.featureVariances[d][j] += Math.pow(index_sample(samples[i], j) - this.featureMeans[d][j], 2);
      }
    }

    for (let i = 0; i < this.n_classes; i++){
      for (let j = 0; j < this.n_features; j++){
        this.featureVariances[i][j] /= this.probs[i] - 1;
      }
    }

    this.probs[0] /= samples.length;
    this.probs[1] /= samples.length;
  }


  classify(features){
    let max_prob = -100000000000;
    let max_label = -1;
    for (let i = 0; i < this.n_classes; i++){
      let prob = 0;
      for (let j = 0; j < this.n_features; j++){
        prob += this.log_prob(index_sample(features, j), this.featureMeans[i][j], this.featureVariances[i][j]);
      }
      prob += Math.log(this.probs[i]);
      if(prob > max_prob)
      {
        max_prob = prob;
        max_label = i;
      }
    }

    return max_label;
  }

  log_prob(x, mean, variance){
    let pr = Math.log(2 * Math.PI * variance);
    pr += ((x - mean) * (x - mean)) / variance;
    return -0.5 * pr;
  }
}

function shuffleData(k, data){
  let sdata = [];
  for (let i = 0; i < k; i++){
    sdata.push(new Array())
  }

  for (let i = 0; i < data.length; i++){
    sdata[i % k].push(data[i])
  }

  return sdata;
}

function grabFeature(d, pos){
  let s = d[pos];
  let amt = Object.keys(s).length;
  let new_s = [];
  for (let i = 0; i < amt; i++){
    if (Object.entries(s)[i][0] != 'diagnosis'){
      new_s.push(Object.entries(s)[i][1]);
    }
  }
  return new_s;
}


function index_sample(sample, i){
  if (i == 0) {
    return sample.skin_thickness;
  } else if (i == 1){
    return sample.pregnancies;
  } else if (i == 2){
    return sample.blood_pressure;
  } else if (i == 3){
    return sample.bmi;
  } else if (i == 4){
    return sample.insulin;
  } else if (i == 5){
    return sample.age;
  } else if (i == 6){
    return sample.glucose;
  } else if (i == 7){
    return sample.diabetes_pedigree_function;
  }
}

function NaiveBayesAverage(amount){
  let avgScores = [];
  let c = 0;
  for (let a = 0; a < amount; a++){
    let buckets = shuffleData(k, shuffleArray(data))
    let scores = [];

    for (let i = 0; i < buckets.length; i++){
      let count = 0;
      let mergedArray = [];
      const naive_bayes = new NaiveBayes(size, 2);

      for (let j = 0; j < buckets.length; j++){
        if (i != j){
          mergedArray = mergedArray.concat(buckets[j]);
        }
      }
      naive_bayes.train(mergedArray);

      for (let index = 0; index < buckets[i].length; index++){
        let result = naive_bayes.classify(buckets[i][index]);
        if (result == buckets[i][index].diagnosis){
          count++;
        }
      }

      count /= buckets[i].length;
      scores.push(count * 100);
    }

    let avgScore = 0;
    for (let i = 0; i < scores.length; i++){
      avgScore += scores[i];

    }

    avgScore /= scores.length;
    avgScores.push(avgScore);
    console.log(a, ': The average score is ', avgScore, '%');
  }


  for (let i = 0; i < avgScores.length; i++){
    c += avgScores[i];
  }

  c /= avgScores.length;
  console.log('The entire average was ', c, '%');
}

NaiveBayesAverage(10);
