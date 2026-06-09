mod body;
mod trajectory;
use crate::body::Body;
use crate::trajectory::{calc_gravity_acceleration, has_collision};
use nalgebra::{Matrix3xX, RowDVector, RowVector3, SVector, Vector3};
use ode_solvers::{Dopri5, System};

/*
x1  y1  z1  x2  y2  z2  x3  y3  z3
vx1 vy1 vz1 vx2 vy2 vz2 vx3 vy3 vz3
--------------------
total: 18
(x,  y,  z ) - position
(vx, vy, vz) - velocity
*/

type State = SVector<f64, 18>;

fn calc_skipper_pos(i: usize) -> usize {
    i * 3
}

fn calc_skipper_vel(i: usize) -> usize {
    (i * 3) + 9
}

pub fn init_state(bodies: &[Body; 3]) -> State {
    let mut state = State::zeros();

    for (i, b) in bodies.iter().enumerate() {
        let skipper_pos = calc_skipper_pos(i);
        let skipper_vel = calc_skipper_vel(i);

        state[skipper_pos] = b.position.x;
        state[skipper_pos + 1] = b.position.y;
        state[skipper_pos + 2] = b.position.z;
        state[skipper_vel] = b.velocity.x;
        state[skipper_vel + 1] = b.velocity.y;
        state[skipper_vel + 2] = b.velocity.z;
    }
    state
}

struct ThreeBodySystem {
    pub masses: RowDVector<f64>,
    pub bodies: Vec<Body>,
}

impl ThreeBodySystem {
    pub fn init(bodies: &[Body]) -> Self {
        let mut mass_vec = RowDVector::<f64>::zeros(bodies.len());
        for (i, b) in bodies.iter().enumerate() {
            mass_vec[i] = b.mass;
        }

        Self {
            masses: mass_vec,
            bodies: bodies.to_vec(),
        }
    }
}

impl System<f64, State> for ThreeBodySystem {
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let n = 3;
        let mut pos = Matrix3xX::<f64>::zeros(n);

        for i in 0..n {
            let body_pos = RowVector3::<f64>::from_row_slice(&y.as_slice()[(i * 3)..(i * 3 + 3)]);
            pos.set_column(i, &body_pos.transpose());
        }

        let acc = calc_gravity_acceleration(&pos, &self.masses);

        for i in 0..n {
            let skipper_pos = calc_skipper_pos(i);
            let skipper_vel = calc_skipper_vel(i);
            // y velocity -> dy position
            dy[skipper_pos] = y[skipper_vel]; // x
            dy[skipper_pos + 1] = y[skipper_vel + 1]; // y
            dy[skipper_pos + 2] = y[skipper_vel + 2]; // z

            // acc -> dy velocity
            dy[skipper_vel] = acc[(i, 0)]; // vx
            dy[skipper_vel + 1] = acc[(i, 1)]; // vy
            dy[skipper_vel + 2] = acc[(i, 2)]; // vz
        }
    }

    fn solout(&mut self, _t: f64, y: &State, _dy: &State) -> bool {
        for i in 0..3 {
            let body_pos = Vector3::<f64>::from_row_slice(&y.as_slice()[(i * 3)..(i * 3 + 3)]);
            // update body position;
            self.bodies[i].position = body_pos;
        }
        has_collision(&self.bodies)
    }
}

fn main() {
    let bodies = [
        Body::new(20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        Body::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        Body::new(-20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
    ];

    let system = ThreeBodySystem::init(&bodies);
    let state = init_state(&bodies);
    let mut stepper: Dopri5<f64, State, ThreeBodySystem> =
        Dopri5::new(system, 0.0, 150.0, 0.002, state, 1.0e-14, 1.0e-14);
    let res = stepper.integrate();

    match res {
        Ok(stats) => {
            println!("stats: {}", stats);
        }
        Err(e) => {
            println!("e: {}", e);
        }
    }
}
