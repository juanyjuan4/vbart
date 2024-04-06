mod registration {
    include!(concat!(env!("OUT_DIR"), "/registration.rs"));
}

use geo::point;
use geo::prelude::*;
use geo::Point;
use rand::prelude::*;
use rand_distr::{Distribution, Gamma, Normal, Poisson, Standard, Uniform};
use roxido::*;
use statrs::function::gamma::gamma;

#[derive(Clone)]
struct Sobj {
    s_mat: Vec<Vec<f64>>,
    dtr: Vec<f64>,
    dtd: Vec<f64>,
    mu_index: Vec<i32>,
    log_crit: f64,
}

#[roxido]
fn vbart_sampler(
    y: &RObject<RVector, f64>,
    n_samples: usize,
    n_trees: usize,
    delta: f64,
    lon: &RObject<RVector, f64>,
    lat: &RObject<RVector, f64>,
    s_list_r: &mut RObject<RList>,
    mu_list_r: &mut RObject<RList>,
    tau_vec_r: &mut RObject<RVector, f64>,
    yhat_mat_r: &mut RObject<RMatrix, f64>,
) {
    let mut rng = rand::thread_rng();
    let nrows = y.len();
    let ymean: f64 = y.slice().iter().fold(0.0, |s, &z| s + z) / nrows as f64;
    let dy = vec![0.0; nrows];

    let mut s_list = Vec::new();
    let mut mu_list = Vec::new();
    for _scan_index in 0..n_samples {
        s_list.push(Vec::new());
        mu_list.push(Vec::new());
    }

    let mut tau_vec = vec![1.0; n_samples];
    let mut tree_contribution = vec![vec![0.0; nrows]; n_trees];

    println!("First check");
    // First element of matrix columns
    for tree_index in 0..n_trees {
        s_list[0].push(sample_s_given_r_tau(
            &vec![0.0; nrows],
            tau_vec[0],
            delta,
            nrows,
            lon,
            lat,
        ));

        let temp_dtr = s_list[0][tree_index].dtr.clone();
        let temp_dtd = s_list[0][tree_index].dtd.clone();
        mu_list[0].push(sample_mu_given_sr_tau(temp_dtr, temp_dtd, tau_vec[0]));

        for i in 0..nrows {
            tree_contribution[tree_index][i] =
                mu_list[0][tree_index][s_list[0][tree_index].mu_index[i] as usize];
        }
    }

    let mut yhat = vec![0.0; nrows];
    for i in 0..nrows {
        let mut cur_sum = 0.0;
        for j in 0..n_trees {
            cur_sum += tree_contribution[j][i];
        }
        yhat[i] = cur_sum;
    }

    let mut yhat_mat = vec![vec![0.0; nrows]; n_samples];
    for i in 0..nrows {
        yhat_mat[1][i] = yhat.get(i).stop() + ymean;
    }

    for scan_index in 1..n_samples {
        rprintln!("Iteration {}", scan_index + 1);
        for tree_index in 0..n_trees {
            let mut r = vec![0.0; nrows];
            for i in 0..n_trees {
                r[i] = yhat[i] + tree_contribution[tree_index][i];
            }

            // Update S conditional on R and tau
            let prop_s = sample_s_given_r_tau(&r, tau_vec[scan_index - 1], delta, nrows, lon, lat);
            if prop_s.log_crit - s_list[scan_index - 1][tree_index].log_crit
                > StdRng::from_rng(&mut rng).unwrap().sample(Standard)
            {
                s_list[scan_index].push(prop_s);
            } else {
                let temp = s_list[scan_index - 1][tree_index].clone();
                s_list[scan_index].push(temp);
            }
            let temp_dtr = s_list[scan_index][tree_index].dtr.clone();
            let temp_dtd = s_list[scan_index][tree_index].dtr.clone();
            mu_list[scan_index].push(sample_mu_given_sr_tau(
                temp_dtr,
                temp_dtd,
                tau_vec[scan_index - 1],
            ));
            let mut new_tree_contribution = vec![0.0; nrows];
            let yhat_old = yhat.clone();
            for i in 0..nrows {
                new_tree_contribution[i] = mu_list[scan_index][tree_index]
                    [s_list[scan_index][tree_index].mu_index[i] as usize];
                yhat[i] = yhat_old.get(i).stop() - tree_contribution[tree_index][i]
                    + new_tree_contribution.get(i).stop();
                tree_contribution[tree_index][i] = new_tree_contribution[i];
            }
        }
        // Update yhat_mat and tau_vec
        for i in 0..nrows {
            yhat_mat[scan_index][i] = yhat.get(i).stop() + ymean;
        }
        tau_vec[scan_index] = sample_tau_given_everything(&yhat, &dy, 2.0, 2.0);
    }
    rprintln!("Finished sampling. Writing to list now");

    // This is where I'm going to populate all of the R data objects that were passed in.
    for scan_index in 0..n_samples {
        for tree_index in 0..n_trees {
            // s_list
            let temp_init = s_list_r
                .get_mut(scan_index)
                .stop()
                .as_list_mut()
                .stop()
                .get_mut(tree_index)
                .stop()
                .as_list_mut()
                .stop();

            let temp: &mut RObject<RMatrix, f64> = temp_init
                .get_mut(0)
                .stop()
                .as_matrix_mut()
                .stop()
                .as_f64_mut()
                .stop();
            for i in 0..s_list[scan_index][tree_index].s_mat.len() {
                for j in 0..2 {
                    temp.set((i, j), s_list[scan_index][tree_index].s_mat[i][j])
                        .stop();
                }
            }

            let temp: &mut RObject<RVector, f64> = temp_init
                .get_mut(1)
                .stop()
                .as_vector_mut()
                .stop()
                .as_f64_mut()
                .stop();
            for i in 0..s_list[scan_index][tree_index].dtr.len() {
                temp.set(i, s_list[scan_index][tree_index].dtr[i]).stop()
            }

            let temp: &mut RObject<RVector, f64> = temp_init
                .get_mut(2)
                .stop()
                .as_vector_mut()
                .stop()
                .as_f64_mut()
                .stop();
            for i in 0..s_list[scan_index][tree_index].dtd.len() {
                temp.set(i, s_list[scan_index][tree_index].dtd[i]).stop()
            }

            let temp: &mut RObject<RVector, i32> = temp_init
                .get_mut(3)
                .stop()
                .as_vector_mut()
                .stop()
                .as_i32_mut()
                .stop();
            for i in 0..s_list[scan_index][tree_index].mu_index.len() {
                temp.set(i, s_list[scan_index][tree_index].mu_index[i])
                    .stop()
            }

            let temp: &mut RObject<RScalar, f64> = temp_init
                .get_mut(4)
                .stop()
                .as_scalar_mut()
                .stop()
                .as_f64_mut()
                .stop();
            temp.set(s_list[scan_index][tree_index].log_crit);

            // mu_list
            let temp: &mut RObject<RVector, f64> = mu_list_r
                .get_mut(scan_index)
                .stop()
                .as_list_mut()
                .stop()
                .get_mut(tree_index)
                .stop()
                .as_vector_mut()
                .stop()
                .as_f64_mut()
                .stop();
            for i in 0..mu_list[scan_index][tree_index].len() {
                temp.set(i, mu_list[scan_index][tree_index][i]).stop();
            }

            // tau_vec
            for i in 0..tau_vec.len() {
                tau_vec_r.set(i, tau_vec[i]).stop();
            }

            // yhat_mat
            for i in 0..yhat_mat.len() {
                for j in 0..yhat_mat[i].len() {
                    yhat_mat_r.set((i, j), yhat_mat[i][j]).stop();
                }
            }
        }
    }

    let ret = RObject::<RScalar, i32>::from_value(1, pc);
    ret
}

fn sample_s_given_r_tau<'a>(
    r: &Vec<f64>,
    tau: f64,
    delta: f64,
    nrow: usize,
    lon: &'a RObject<RVector, f64>,
    lat: &'a RObject<RVector, f64>,
) -> Sobj {
    let mut rng = rand::thread_rng();
    let poi = Poisson::new(delta).unwrap();
    let b = poi.sample(&mut rng) as usize;
    let uno = Uniform::new(-111.2, -109.3);
    let dos = Uniform::new(40.5, 41.0);
    let mut s_mat = vec![vec![0.0; 2]; b + 1];
    for i in 0..(b + 1) {
        s_mat[i][0] = uno.sample(&mut rng);
        s_mat[i][1] = dos.sample(&mut rng);
    }
    // rprintln!("{:?}", s_mat);

    let mut d_mat = vec![vec![0.0; b + 1]; nrow];
    for i in 0..nrow {
        for j in 0..b {
            d_mat[i][j] = euc_dist(
                point!(x: lon.get(i).stop(), y: lat.get(i).stop()),
                point!(x: s_mat[j][0], y: s_mat[j][1]),
            );
        }
    }

    let mut mu_index = vec![0; nrow];
    // Find the row minimum
    for i in 0..nrow {
        let mut cur_min = d_mat[i][0];
        let mut cur_min_index: i32 = 0;
        for j in 1..b {
            let ii = d_mat[i][j];
            if ii < cur_min {
                cur_min = ii;
                cur_min_index = j as i32;
            }
        }
        mu_index[i] = cur_min_index;
    }

    let mut dtr = vec![0.0; b + 1];
    let mut dtd = vec![0.0; b + 1];

    for i in 0..b {
        for j in 0..nrow {
            let mut dtr_temp = 0.0;
            let mut dtd_temp = 0.0;
            if i == mu_index[j] as usize {
                dtr_temp += r.get(j).stop();
                dtd_temp += 1.0;
            }
            dtr[i] = dtr_temp;
            dtd[i] = dtd_temp;
        }
    }

    let log_crit_temp: f64 = dtr
        .iter()
        .zip(dtd.iter())
        .map(|(x, y)| (-(1.0 + tau * y).ln() + tau.powi(2) / (1.0 + tau * y) * x.powi(2)) as f64)
        .sum();
    let log_crit: f64 = log_crit_temp / 2.0 + pois_pmf(b as i32, delta).ln();
    // let mut s_new = vec![vec![0.0; 2]; b + 1];
    // for i in 0..(b + 1) {
    //     for j in 0..2 {
    //         s_new[i][j] = s_mat[i][j];
    //     }
    // }
    let ret = Sobj {
        s_mat,
        dtr,
        dtd,
        mu_index,
        log_crit,
    };
    ret
}

fn pois_pmf(k: i32, lambda: f64) -> f64 {
    if k < 0 {
        0.
    } else {
        lambda.powi(k) * (-lambda).exp() / gamma(k as f64)
    }
}

fn euc_dist(p1: Point<f64>, p2: Point<f64>) -> f64 {
    p1.haversine_distance(&p2)
}

fn sample_mu_given_sr_tau<'a>(dtr: Vec<f64>, dtd: Vec<f64>, tau: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut ret = vec![0.0; dtd.len()];
    for i in 0..dtd.len() {
        let norm = Normal::new(
            tau * dtr.get(0).stop() / (1.0 + tau * dtd.get(0).stop()),
            1.0 / (1.0 + tau * dtd.get(0).stop()).sqrt(),
        )
        .unwrap();
        ret[i] = norm.sample(&mut rng);
    }
    ret
}

fn sample_tau_given_everything(yhat: &Vec<f64>, dy: &Vec<f64>, a: f64, b: f64) -> f64 {
    let mse: f64 = dy
        .iter()
        .zip(yhat.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    let gam = Gamma::new(a + (yhat.len() as f64) / 2.0, b + mse / 2.0).unwrap();
    let mut rng = rand::thread_rng();
    gam.sample(&mut rng)
}
