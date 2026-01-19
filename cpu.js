function runACO() {
  const n = cities.length;
  const ants = 40;
  const iterations = 200;
  const decay = 0.5;
  const alpha = 1;
  const beta = 3;

  const pheromone = Array.from({length: n}, () => Array(n).fill(1));
  const distMatrix = Array.from({length: n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      distMatrix[i][j] = i === j ? 1e-10 : distance(cities[i], cities[j]);

  function pickNext(city, visited) {
    const probs = [];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (visited.has(i)) { probs[i] = 0; continue; }
      const p = Math.pow(pheromone[city][i], alpha) *
                Math.pow(1/distMatrix[city][i], beta);
      probs[i] = p;
      sum += p;
    }
    const r = Math.random() * sum;
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += probs[i];
      if (acc >= r) return i;
    }
    return probs.indexOf(Math.max(...probs));
  }

  let bestPath = null;
  let bestLen = Infinity;

  for (let it = 0; it < iterations; it++) {
    const paths = [];

    for (let a = 0; a < ants; a++) {
      const visited = new Set();
      const path = [];
      let city = Math.floor(Math.random() * n);
      path.push(city);
      visited.add(city);

      while (path.length < n) {
        city = pickNext(city, visited);
        path.push(city);
        visited.add(city);
      }

      const len = pathLength(path, cities);
      paths.push({path, len});
      if (len < bestLen) {
        bestLen = len;
        bestPath = path;
      }
      paths.push({path, len});
    }

    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        pheromone[i][j] *= (1 - decay);

    for (const {path, len} of paths) {
      const add = 1/len;
      for (let i = 0; i < path.length; i++) {
        const c1 = path[i];
        const c2 = path[(i+1) % path.length];
        pheromone[c1][c2] += add;
        pheromone[c2][c1] += add;
      }
    }
  }

  draw(bestPath, cities);
  document.getElementById("out").textContent =
    `ACO длина маршрута: ${bestLen.toFixed(2)}`;
}