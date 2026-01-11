const gpu = new GPU();

function solveTSP(distances,Ants=1000,courierCount=1,iterations=15,alpha=1,beta=5,evaporationRate=0.5) {

    const N = distances.length;
    const limit = Math.floor(N / courierCount);

    const selectCity = gpu.createKernel(function (pozitions,distances, pheromones,visited,curNum) {
        const N = this.constants.N;
        const alpha = this.constants.alpha;
        const beta = this.constants.beta;
        const ant = this.thread.x;
        const cur_poz = pozitions[ant];
        let total = 0;
        for (let city = 0; city < N; city++) {
            const distance = distances[cur_poz][city];
            const visit = visited[ant][city];
            if (distance == 0 || visit == 1) {
                continue;
            }
            const pheromone = pheromones[cur_poz*curNum][city];
            total += Math.pow(pheromone, alpha) * Math.pow(1 / distance, beta);
        }
        let random = Math.random() * total;
        for (let city = 0; city < N; city++) {
            const distance = distances[cur_poz][city];
            const visit = visited[ant][city];
            if (distance == 0 || visit == 1) {
                continue;
            }
            const pheromone = pheromones[cur_poz*curNum][city];
            random -= Math.pow(pheromone, alpha) * Math.pow(1 / distance, beta);
            if (random <= 0) {
                return city;
            }
        }
        return 0;
    },{
        output: [Ants],
        constants: {N,alpha,beta},
        pipeline: true,
        immutable: true
    });
    const updateVisited = gpu.createKernel(function (pozitions,visited) {
        const ant   = this.thread.y;
        const city  = this.thread.x;
        let visit   = visited[ant][city];
        if (pozitions[ant] == city) {
            visit = 1;    
        };
        return visit;
    },{
        output: [N,Ants],
        pipeline: true,
        immutable: true
    });
    const updateRoads = gpu.createKernel(function (pozitions,roads,cur_step,curNum,min,max) {
        const Ants = this.constants.Ants;
        const ant   = this.thread.y;
        const step     = this.thread.x;
        const cur_ant = ant * curNum;
        let city    = roads[cur_ant][step];
        if (step == cur_step && cur_ant >= min && cur_ant <= max) {
            city = pozitions[ant];    
        };
        return city;
    },{
        output: [limit+1,Ants*courierCount],
        constants: {Ants},
        pipeline: true,
        immutable: true
    });
    const updatePheromones = gpu.createKernel(function (pheromones) {
        const evaporationRate     = this.constants.evaporationRate;
        const x     = this.thread.x;
        const y     = this.thread.y;
        return pheromones[x][y] * (1 - evaporationRate);
    },{
        output: [N,N],
        constants: {evaporationRate},
        pipeline: true,
        immutable: true
    });
    const calcDistances = gpu.createKernel(function (roads,distances) {
        const ant   = this.thread.x;
        const N     = this.constants.N;
        let length      = 0;
        for (let step = 0; step < N-1; step++) {
            const a = roads[ant][step]; 
            const b = roads[ant][step+1];
            length += distances[a][b];
        }
        return length;
    },{
        output: [Ants],
        constants: {N},
        pipeline: true,
        immutable: true
    });

    let bestPath = null;
    let bestLength = Infinity;


    let pheromones  = new Array(N*courierCount).fill(0).map(() => new Array(N).fill(0.1));

    const empty_pozitions   = new Array(Ants).fill(0);
    const empty_visited     = new Array(Ants).fill(0).map(() => {const array = new Array(N).fill(0); array[0] = 1; return array});
    const empty_roads       = new Array(Ants*courierCount).fill(0).map(() => new Array(limit+1).fill(0));

    for (let i = 0; i < iterations; i++) {
        let pozitions     = structuredClone(empty_pozitions);
        let visited       = structuredClone(empty_visited);
        let roads         = structuredClone(empty_roads);
        let new_pozitions = undefined;
        let new_visited   = undefined;
        let new_roads     = undefined;
        let isFirst = true;
        let currier = 0;

        let cur_step = 0;
        let curNum = 1;
        for (let step = 1; step < N; step++) {
            if (cur_step > limit) {
                cur_step = 0; 
                curNum += 1;   
            }
            cur_step += 1;

            new_pozitions   = selectCity(pozitions,distances, pheromones,visited,curNum);
            new_visited     = updateVisited(new_pozitions,visited);
            let min = 0;
            let max = 99;
            new_roads       = updateRoads(new_pozitions,roads,cur_step,curNum,min,max);
            if (!isFirst) {
                pozitions.delete();
                visited.delete();
                roads.delete();
            }  
            isFirst = false;
            pozitions = new_pozitions;
            visited   = new_visited;
            roads     = new_roads;
        }
        pozitions.delete();
        visited.delete();

        const new_pheromones = updatePheromones(pheromones);
        pheromones  = new_pheromones.toArray(); 
        new_pheromones.delete();

        new_roads = roads.toArray();
        roads.delete();
        roads = new_roads;

        const antlength  = calcDistances(roads,distances).toArray();
        for (let ant = 0; ant < Ants; ant++) {
            const lengt = antlength[ant];
            const pheromoneContribution = 1000 / lengt;
            for (let step = 0; step < N-1; step++) {
                const a = roads[ant][step]; 
                const b = roads[ant][step+1];
                pheromones[a][b] += pheromoneContribution;
            }
            if (lengt < bestLength) {
                bestLength = lengt;
                bestPath = roads[ant];
            }
        }
        console.log(bestLength);   
    }

    return {bestLength,bestPath};
}

function generateDistanceMatrix(size) {
    const matrix = new Array(size).fill(0).map(() => new Array(size).fill(0));
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (i !== j) {
                matrix[i][j] = Math.floor(Math.random() * 100) + 1;
            }
        }
    }
    return matrix;
}

const distances = generateDistanceMatrix(50);
const result    = solveTSP(distances,100,2,10);
console.log(result);
