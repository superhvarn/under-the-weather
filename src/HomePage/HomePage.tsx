import "./HomePage.css"
// import 'bootstrap/dist/css/bootstrap.min.css';

import App from "../App";
import {useState} from "react";
export const HomePage = (props: any) => {
    const [currentPage, setCurrentPage] = useState('home');

    const togglePage = (pageName: string) => {
        setCurrentPage(pageName)
    }

    const handleLogin = (e: any) => {
        e.preventDefault();
        props.onFormSwitch('login');
    }

    const handleReg = (e: any) => {
        e.preventDefault();
        props.onFormSwitch('register');
    }

    return (
        <>
            <h1>Under the Weather</h1>
            <div className={"d-flex"}>
                <div className={"d-flex flex-row"}>
                    <button onClick={handleLogin}>Log In</button>
                    <button onClick={handleReg}>Register</button>
                </div>
            </div>
        </>
    )
}