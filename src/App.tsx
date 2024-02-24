import React, {useState} from 'react';
import {Register} from "./Authentification/Register"
import {Login} from "./Authentification/Login"
import {ZipCode} from "./Authentification/ZipCode";
import './Authentification/authentication.css'
import {HomePage} from "./HomePage/HomePage";

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const togglePage = (pageName: string) => {
    setCurrentPage(pageName)
  }

  return (
      <div className="App">
          <header className="App-header">
              {currentPage === "login" ? (
                  <Login onFormSwitch={togglePage}/>
              ) : currentPage === "zipcode" ? (
                  <ZipCode onFormSwitch={togglePage}/>
              ) : currentPage === "register" ? (
                  <Register onFormSwitch={togglePage}/>
              ) : (
                  <HomePage onFormSwitch={togglePage}/>
              )
              }
          </header>
      </div>
  );
}

export default App;