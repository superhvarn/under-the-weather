import "./HomePage.css";
// import 'bootstrap/dist/css/bootstrap.min.css';

import { useState } from "react";

export const HomePage = (props : any) => {
    const [currentPage, setCurrentPage] = useState('home');

    const togglePage = (pageName : any) => {
        setCurrentPage(pageName);
    };

    const handleLogin = (e : any) => {
        e.preventDefault();
        props.onFormSwitch('login');
    };

    const handleReg = (e : any) => {
        e.preventDefault();
        props.onFormSwitch('register');
    };

    return (
        <div>
            <img src={`${process.env.PUBLIC_URL}/logo.png`} width={250} height={250} alt="Logo" className="homepage-logo"/>
            <h1>Under the Weather</h1>
            <div className={"d-flex"}>
                <div className={"d-flex flex-row"}>
                    <button onClick={handleLogin}>Log In</button>
                    <button onClick={handleReg}>Register</button>
                </div>
            </div>
        </div>
    );
}
