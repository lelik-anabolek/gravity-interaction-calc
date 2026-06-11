mod body;
mod solver;
mod trajectory;
use crate::body::Body;
use crate::solver::integrate;

fn main() {
    let bodies = [
        Body::new(20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        Body::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        Body::new(-20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
    ];

    let res = integrate(&bodies, None, None);

    match res {
        Ok(stats) => {
            println!("stats: {:?}", stats);
        }
        Err(e) => {
            println!("e: {}", e);
        }
    }
}
