import { useEffect, useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove, preload } from './engine/alphazero/index.js'
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
  const [thinking, setThinking] = useState(false)

  // Warm up the ONNX session as soon as the page loads, so the first bot move
  // doesn't pay the model load.
  useEffect(() => {
    preload().catch((err) => console.warn('AlphaZero preload failed', err))
  }, [])

  // The bot searches asynchronously: show the human's move right away, then
  // apply the bot's reply when it's ready.
  async function handleMove(move) {
    if (thinking || getTurn(game) !== HUMAN || isGameOver(game)) return
    const next = applyMove(game, move)
    if (!next) return
    setGame(next)
    if (isGameOver(next)) return

    setThinking(true)
    try {
      setGame(applyMove(next, await bestMove(next)))
    } finally {
      setThinking(false)
    }
  }

  const interactive = !thinking && getTurn(game) === HUMAN && !isGameOver(game)

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
      <button className="new-game" onClick={() => setGame(createGame())} disabled={thinking}>
        New Game
      </button>
    </main>
  )
}
