//! Rules parity with src/engine/game.js. Replays the games dumped by
//! scripts/dump_rules_fixture.mjs and checks, at every ply, that the Rust board
//! has the same state, legal pawn moves and legal walls as the JS engine, and
//! that both agree on the winner.
//!
//! Generate the fixture first (from the repo root):
//!   node scripts/dump_rules_fixture.mjs
//! Override its path with QUORIDOR_RULES_FIXTURE.

use quoridor_core::board::{Board, Move};
use serde_json::Value;

fn fixture_path() -> String {
    std::env::var("QUORIDOR_RULES_FIXTURE")
        .unwrap_or_else(|_| format!("{}/../fixtures/rules.json", env!("CARGO_MANIFEST_DIR")))
}

fn sq((r, c): (usize, usize)) -> String {
    format!("{r},{c}")
}

fn wall(horiz: bool, r: usize, c: usize) -> String {
    format!("{}{r},{c}", if horiz { 'h' } else { 'v' })
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

fn parse_move(s: &str) -> Move {
    let (kind, rest) = s.split_at(1);
    let (r, c) = rest.split_once(',').unwrap();
    let (r, c) = (r.parse().unwrap(), c.parse().unwrap());
    match kind {
        "p" => Move::Pawn { r, c },
        "h" => Move::Wall { horiz: true, r, c },
        _ => Move::Wall { horiz: false, r, c },
    }
}

#[test]
fn matches_js_rules() {
    let path = fixture_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run scripts/dump_rules_fixture.mjs)"));
    let games: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut n_plies = 0;
    let (mut n_jumps, mut n_diagonals) = (0, 0);

    for (gi, game) in games.iter().enumerate() {
        let mut b = Board::new();
        for (pi, ply) in game["plies"].as_array().unwrap().iter().enumerate() {
            let at = format!("game {gi} ply {pi}");
            let turn = b.turn();
            assert_eq!(ply["turn"].as_u64().unwrap() as usize, turn, "{at}: turn");
            assert_eq!(strs(&ply["pawns"]), vec![sq(b.pawn(0)), sq(b.pawn(1))], "{at}: pawns");
            let left: Vec<usize> =
                ply["wallsLeft"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect();
            assert_eq!(left, vec![b.walls_left(0), b.walls_left(1)], "{at}: walls left");
            let mut placed = Vec::new();
            for horiz in [true, false] {
                for r in 0..8 {
                    for c in 0..8 {
                        if b.has_wall(horiz, r, c) {
                            placed.push(wall(horiz, r, c));
                        }
                    }
                }
            }
            assert_eq!(strs(&ply["walls"]), sorted(placed), "{at}: walls");

            let pawn = sorted(b.legal_pawn_targets().into_iter().map(sq).collect());
            assert_eq!(strs(&ply["pawn"]), pawn, "{at}: legal pawn moves");
            let walls = sorted(b.legal_walls().into_iter().map(|(h, r, c)| wall(h, r, c)).collect());
            assert_eq!(strs(&ply["legalWalls"]), walls, "{at}: legal walls");

            // Every legal move must round-trip through its policy index.
            let n_legal = b.legal_moves().len();
            assert_eq!(n_legal, pawn.len() + walls.len(), "{at}: legal move count");
            let (me, opp) = (b.pawn(turn), b.pawn(1 - turn));
            for (r, c) in b.legal_pawn_targets() {
                let dist = me.0.abs_diff(r) + me.1.abs_diff(c);
                let adjacent = me.0.abs_diff(opp.0) + me.1.abs_diff(opp.1) == 1;
                if adjacent && dist == 2 {
                    if me.0 == r || me.1 == c {
                        n_jumps += 1;
                    } else {
                        n_diagonals += 1;
                    }
                }
            }

            let m = parse_move(ply["move"].as_str().unwrap());
            let idx = b.move_index(m);
            assert!(b.legal_moves().contains(&idx), "{at}: move {m:?} not legal in Rust");
            assert_eq!(b.decode(idx), m, "{at}: decode");
            b.push(idx);
            n_plies += 1;
        }
        let winner = game["winner"].as_u64().map(|w| w as usize);
        assert_eq!(b.winner(), winner, "game {gi}: winner");
    }
    println!("{} games, {n_plies} plies, {n_jumps} jumps, {n_diagonals} diagonals", games.len());
    assert!(n_jumps > 0 && n_diagonals > 0, "fixture never exercised jumps/diagonals");
}
