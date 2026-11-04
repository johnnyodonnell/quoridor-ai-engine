import { useState } from 'react'
import { BOT, isLegalWall, legalPawnMoves, wallList } from '../engine/game.js'

// SVG Quoridor board. Row 0 (the human's home edge) is drawn at the bottom.
// Click a highlighted square to move; hover a groove to preview a wall and
// click to place it. `interactive` (computed by the caller from the live game)
// controls whether input is accepted. onMove(move) receives a game.js move.

const CELL = 48
const GAP = 12
const PITCH = CELL + GAP
const SIZE = 9 * CELL + 8 * GAP

const clamp = (n) => Math.max(0, Math.min(7, n))
const cellX = (c) => c * PITCH
const cellY = (r) => (8 - r) * PITCH

function wallRect({ r, c, o }) {
  if (o === 'h') {
    return { x: cellX(c), y: cellY(r + 1) + CELL, width: 2 * CELL + GAP, height: GAP }
  }
  return { x: cellX(c) + CELL, y: cellY(r + 1), width: GAP, height: 2 * CELL + GAP }
}

// Maps a pointer position (in SVG units) to a square or a wall anchor. A wall
// is centred on the groove intersection nearest the pointer.
function hitTest(game, x, y) {
  if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) return null
  const col = Math.floor(x / PITCH)
  const row = Math.floor(y / PITCH)
  const inCellX = x - col * PITCH < CELL
  const inCellY = y - row * PITCH < CELL
  if (inCellX && inCellY) return { kind: 'square', r: 8 - row, c: col }

  const nearC = clamp(Math.round((x + GAP / 2) / PITCH) - 1)
  const nearR = clamp(Math.round((SIZE - y + GAP / 2) / PITCH) - 1)
  let candidates
  if (inCellX) candidates = [{ r: 7 - row, c: nearC, o: 'h' }]
  else if (inCellY) candidates = [{ r: nearR, c: col, o: 'v' }]
  else candidates = [{ r: 7 - row, c: col, o: 'h' }, { r: 7 - row, c: col, o: 'v' }]
  const wall = candidates.find((w) => isLegalWall(game, w))
  return wall ? { kind: 'wall', wall } : null
}

export default function Board({ game, interactive, onMove }) {
  const [ghost, setGhost] = useState(null)
  const targets = interactive ? legalPawnMoves(game) : []
  const isTarget = (r, c) => targets.some((m) => m.r === r && m.c === c)

  const last = game.lastMove?.player === BOT ? game.lastMove : null
  const lastSquares =
    last?.move.type === 'pawn' ? [last.from, { r: last.move.r, c: last.move.c }] : []
  const isLastSquare = (r, c) => lastSquares.some((s) => s.r === r && s.c === c)
  const isLastWall = (w) =>
    last?.move.type === 'wall' && last.move.r === w.r && last.move.c === w.c && last.move.o === w.o

  function pointerHit(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const scale = SIZE / rect.width
    return hitTest(game, (e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale)
  }

  function handlePointerMove(e) {
    if (!interactive) return setGhost(null)
    const hit = pointerHit(e)
    setGhost(hit?.kind === 'wall' ? hit.wall : null)
  }

  function handleClick(e) {
    if (!interactive) return
    const hit = pointerHit(e)
    if (hit?.kind === 'square' && isTarget(hit.r, hit.c)) {
      onMove({ type: 'pawn', r: hit.r, c: hit.c })
    } else if (hit?.kind === 'wall') {
      onMove({ type: 'wall', ...hit.wall })
    }
    setGhost(null)
  }

  const squares = []
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      let cls = 'square'
      if (isTarget(r, c)) cls += ' target'
      else if (isLastSquare(r, c)) cls += ' last'
      squares.push(
        <rect key={`${r},${c}`} className={cls} x={cellX(c)} y={cellY(r)} width={CELL} height={CELL} rx={4} />,
      )
    }
  }

  return (
    <svg
      className={`board${interactive ? ' interactive' : ''}`}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      onPointerMove={handlePointerMove}
      onPointerLeave={() => setGhost(null)}
      onClick={handleClick}
    >
      {squares}
      {wallList(game).map((w) => (
        <rect
          key={`${w.o}${w.r},${w.c}`}
          className={`wall p${w.player}${isLastWall(w) ? ' last' : ''}`}
          rx={3}
          {...wallRect(w)}
        />
      ))}
      {ghost && <rect className="wall ghost" rx={3} {...wallRect(ghost)} />}
      {game.pawns.map((p, i) => (
        <circle key={i} className={`pawn p${i}`} cx={cellX(p.c) + CELL / 2} cy={cellY(p.r) + CELL / 2} r={CELL * 0.34} />
      ))}
    </svg>
  )
}
