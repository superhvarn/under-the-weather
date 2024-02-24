import React, {useState} from 'react';
import {Register} from "./Register"
import {Login} from "./Login"
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('register');
  const togglePage = (pageName: string) => {
    setCurrentPage(pageName)
  }

  return (
    <div className="App">
      <header className="App-header">
          {
              currentPage === "login" ? <Login onFormSwitch={togglePage}/> : <Register onFormSwitch={togglePage}/>
          }
      </header>
    </div>
  );
}

export default App;
