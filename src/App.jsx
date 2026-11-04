import { useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove } from './engine/random.js'
import { BOT, HUMAN, applyMove, createGame, getStatus, getTurn, isGameOver } from './engine/game.js'

// Only shown once the game is over; there's no turn indicator.
function statusMessage(game) {
  const status = getStatus(game)
  if (status.kind === 'win') return status.winner === HUMAN ? 'You win' : 'Bot wins'
  return ''
}

export default function App() {
  // Game state is immutable; each move produces a new state object.
  const [game, setGame] = useState(createGame)

  // The random bot is synchronous and instant, so the human's move and the
  // bot's reply land in a single state update.
  function handleMove(move) {
    if (getTurn(game) !== HUMAN || isGameOver(game)) return
    let next = applyMove(game, move)
    if (!next) return
    if (!isGameOver(next)) next = applyMove(next, bestMove(next))
    setGame(next)
  }

  const interactive = getTurn(game) === HUMAN && !isGameOver(game)

  return (
    <main className="app">
      <div className="play">
        <Board game={game} interactive={interactive} onMove={handleMove} />
        <aside className="side">
          <p className="walls-left p1">Bot: {game.wallsLeft[BOT]} walls</p>
          <Status message={statusMessage(game)} />
          <p className="walls-left p0">You: {game.wallsLeft[HUMAN]} walls</p>
        </aside>
      </div>
      <button className="new-game" onClick={() => setGame(createGame())}>
        New Game
      </button>
    </main>
  )
}
