mod body;
mod trajectory;
use crate::body::body::Body;
use crate::trajectory::trajectory::calc_gravity_acceleration;
use nalgebra::{Matrix3, RowVector3, SVector};
use ode_solvers::{Dopri5, System};

/*
x1 y1 z1 vx1 vy1 vz1
x2 y2 z2 vx2 vy2 vz2
x3 y3 z3 vx3 vy3 vz3
--------------------
total: 18
(x, y, z) - position
(x, y, z) - velocity
*/

type State = SVector<f64, 18>;

pub fn init_state(bodies: &[Body; 3]) -> State {
    let mut state = State::zeros();

    for (i, b) in bodies.iter().enumerate() {
        let skipper = 6 * i;
        state[skipper + 0] = b.position.x;
        state[skipper + 1] = b.position.y;
        state[skipper + 2] = b.position.z;
        state[skipper + 3] = b.velocity.x;
        state[skipper + 4] = b.velocity.y;
        state[skipper + 5] = b.velocity.z;
    }

    state
}

struct ThreeBodySystem {
    pub mass_mat: Matrix3<f64>,
}

impl ThreeBodySystem {
    pub fn init(bodies: &[Body; 3]) -> Self {
        let mut mass_vec = RowVector3::<f64>::zeros();
        for (i, b) in bodies.iter().enumerate() {
            mass_vec[i] = b.mass;
        }

        let mass_mat = mass_vec.transpose() * RowVector3::<f64>::from_element(1.0);

        Self { mass_mat }
    }
}

impl System<f64, State> for ThreeBodySystem {
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let n = 3;
        let mut pos = Matrix3::<f64>::zeros();

        for i in 0..n {
            pos[(i, i + 0)] = y[i + 0];
            pos[(i, i + 1)] = y[i + 1];
            pos[(i, i + 2)] = y[i + 2];
        }

        let acc = calc_gravity_acceleration(&pos, &self.mass_mat);

        for i in 0..n {
            let base = 6 * i;
            // y velocity -> dy position
            dy[base + 0] = y[base + 3];
            dy[base + 1] = y[base + 4];
            dy[base + 2] = y[base + 5];
            // acc -> dy velocity
            dy[base + 3] = acc[(i, 0)];
            dy[base + 4] = acc[(i, 1)];
            dy[base + 5] = acc[(i, 2)];
        }
    }
}

fn main() {
    let bodies = [
        Body::new(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        Body::new(-1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        Body::new(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0),
    ];

    let system = ThreeBodySystem::init(&bodies);
    let state = init_state(&bodies);
    let mut stepper: Dopri5<f64, State, ThreeBodySystem> =
        Dopri5::new(system, 0.0, 150.0, 0.002, state, 1.0e-14, 1.0e-14);
    let res = stepper.integrate();
}
