import React, { useState, useEffect } from 'react';
import { Register } from "./Authentification/Register"
import { Login } from "./Authentification/Login"
import { ZipCode } from "./Authentification/ZipCode";
import './Authentification/authentication.css'
import { HomePage } from "./HomePage/HomePage";

function App() {
  // Initialize currentPage from localStorage if available, otherwise default to 'home'
  const [currentPage, setCurrentPage] = useState(sessionStorage.getItem('currentPage') || 'home');

  useEffect(() => {
    // Update localStorage whenever currentPage changes
    sessionStorage.setItem('currentPage', currentPage);
  }, [currentPage]);

  const togglePage = (pageName : any) => {
    setCurrentPage(pageName);
  }

  return (
    <div className="App">
      <header className="App-header">
        {currentPage === "login" ? (
          <Login onFormSwitch={togglePage} />
        ) : currentPage === "zipcode" ? (
          <ZipCode onFormSwitch={togglePage} />
        ) : currentPage === "register" ? (
          <Register onFormSwitch={togglePage} />
        ) : (
          <HomePage onFormSwitch={togglePage} />
        )}
      </header>
    </div>
  );
}

export default App;
