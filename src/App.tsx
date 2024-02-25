import React, { useState, useEffect } from 'react';
import { Register } from "./Authentication/Register"
import { Login } from "./Authentication/Login"
import { ZipCode } from "./Authentication/ZipCode";
import './Authentication/authentication.css'
import HomePage from "./HomePage/HomePage";

function App() {
  // Initialize currentPage from localStorage if available, otherwise default to 'home'
  const [currentPage, setCurrentPage] = useState(localStorage.getItem('currentPage') || 'home');

  useEffect(() => {
    // Update localStorage whenever currentPage changes
    localStorage.setItem('currentPage', currentPage);
  }, [currentPage]);

  const togglePage = (pageName : any) => {
    setCurrentPage(pageName);
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
                ) : null
                }
            </header>
            {/* Render HomePage outside of the header */}
            {currentPage === 'home' && <HomePage onFormSwitch={togglePage} />}
        </div>
    );
}

export default App;
